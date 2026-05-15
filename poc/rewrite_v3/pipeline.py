"""External-calibrated rewrite pipeline V3 entrypoint."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from detect.run import DetectionRunner
from llm.gateway import LLMConfig, LLMGateway
from report.pdf import render_pdf
from report.render_rewrite import render_rewrite_report
from report.report import ReportBuilder, report_to_dict
from rewrite.guards import check_semantic_drift
from rewrite_v2.contracts import build_rewrite_contract
from rewrite_v2.central_module import build_contextual_judgment_plan, score_candidate_against_central_profile
from rewrite_v2.goal_contract import RewriteGoalStatus, evaluate_rewrite_goal
from rewrite_v2.pipeline import _badge_ai, _badge_wq, _extract_original_text, _sentence_comparison
from rewrite_v2.selection import CandidateLane, decide_candidate

from .anchor_validation import validate_v3_candidate
from .assisted_footprint_executor import (
    apply_assisted_footprint_replacements,
    build_assisted_footprint_prompt,
    group_assisted_footprint_windows,
)
from .authorship_window_gate import select_authorship_window_targets
from .candidate_loop import CandidateAction, CandidateIssue, LoopDecision, decide_next_action, issues_from_trace, select_candidate_index
from .compression_policy import compression_policy_for_family, compression_status
from .document_units import compose_units, document_units, word_count
from .external_proxy import evaluate_external_proxy
from .layers.boundary_adapter import build_boundary_adapter_prompt
from .layers.cited_practice_voice import build_cited_practice_voice_chunk_prompt, build_cited_practice_voice_prompt
from .layers.clean_texture_boundary import (
    build_clean_texture_boundary_chunk_prompt,
    build_clean_texture_boundary_prompt,
)
from .layers.contract_repair import build_contract_repair_prompt
from .layers.contrast_boundary import build_contrast_boundary_prompt, extract_contrast_boundary_output
from .layers.document_rhythm import build_document_rhythm_chunk_prompt, build_document_rhythm_prompt
from .layers.detector_ownership_fusion import (
    build_detector_ownership_fusion_prompt,
    extract_fused_document,
    extract_fused_document_with_diagnostics,
)
from .layers.plain_reasoning_broad_prose import build_plain_reasoning_broad_prose_prompt
from .layers.recovery_revision import build_recovery_revision_prompt
from .layers.structure_repair import build_structure_repair_prompt
from .layers.authorship_window_repair import (
    apply_authorship_window_replacements,
    build_authorship_window_repair_prompt,
    extract_authorship_window_replacements,
)
from .output_cleaning import clean_v3_candidate_output
from .portfolio import select_portfolio_candidate
from .paragraph_portfolio_executor import (
    generate_paragraph_ownership_candidate as _run_paragraph_ownership_candidate,
    generate_paragraph_portfolio_candidate as _run_paragraph_portfolio_candidate,
    paragraph_portfolio_config,
)
from .router import route_from_scan_contract
from .scanner_controlled_executor import (
    ScannerControlledConfig,
    build_scanner_controlled_prompt,
    parse_scanner_controlled_variants_with_diagnostics,
    rank_scanner_target_groups,
    restore_protected_anchor_placeholders,
    scanner_controlled_candidate_quality,
    scanner_controlled_metrics,
    scanner_controlled_rank,
    scanner_controlled_variant_gate,
)
from .prompt_contract import group_action_contract
from .scanner_contract import RewriteRiskClass, ScanContract, build_scan_contract, predictability_briefs_from_report
from .style_library import examples_for_family
from .strategy_plan import build_strategy_plan
from .target_executor import (
    SUPPORTED_TARGET_OPERATIONS,
    apply_target_replacements,
    batch_target_groups,
    build_target_executor_prompt,
    group_rewrite_targets,
    missing_required_protected_anchors,
    parse_target_replacements,
    target_execution_trace,
)
from .unit_preserving_prune_bridge import (
    apply_prune_bridge_replacements,
    build_prune_bridge_prompt,
    filter_prune_bridge_groups,
    parse_prune_bridge_replacements,
)

logger = logging.getLogger(__name__)


def _scan_report(text: str) -> dict[str, Any]:
    detect_report = DetectionRunner().run_all(text)
    builder = ReportBuilder()
    builder.add_detection_report(detect_report)
    if detect_report.postprocess_results:
        builder.add_postprocess_results(detect_report.postprocess_results)
    builder.set_meta(scan_time=0, original_text=text)
    payload = report_to_dict(builder.build())
    payload["input_text"] = text
    return payload


def _text_hash(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _predictability_cache_info(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    candidates = [
        report.get("predictability_cache"),
        ((report.get("predictability") or {}).get("predictability_cache") if isinstance(report.get("predictability"), dict) else None),
        (((report.get("scan_intelligence") or {}).get("predictability") or {}).get("predictability_cache") if isinstance(report.get("scan_intelligence"), dict) else None),
        ((((report.get("scan_intelligence") or {}).get("document") or {}).get("predictability") or {}).get("predictability_cache") if isinstance(report.get("scan_intelligence"), dict) else None),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return dict(candidate)
    return {}


def _topk(report: dict | None) -> float | None:
    value = (((report or {}).get("ai_risk_badge") or {}).get("ai_components") or {}).get("topk_pattern_raw")
    return float(value) if isinstance(value, (int, float)) else None


def _ai_footprint_profile(report: dict | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    candidates = [
        report.get("ai_footprint_profile"),
        (((report.get("scan_intelligence") or {}).get("document") or {}).get("ai_footprint_profile")),
        ((report.get("scan_intelligence") or {}).get("ai_footprint_profile")),
        ((report.get("authorship_window_profile") or {}).get("ai_footprint_profile")),
        ((((report.get("scan_intelligence") or {}).get("document") or {}).get("authorship_window_profile") or {}).get("ai_footprint_profile")),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    badge = report.get("ai_risk_badge") if isinstance(report.get("ai_risk_badge"), dict) else {}
    ai_score = _number(badge.get("ai_likelihood_score"), -1.0)
    if ai_score >= 0.0:
        assisted = max(0.0, min(1.0, ai_score / 100.0))
        return {
            "schema_version": "ai_footprint_profile.v2_badge_adapter",
            "basis": "badge_ai_likelihood",
            "fraction_ai": 0.0,
            "fraction_ai_assisted": round(assisted, 4),
            "fraction_human": round(max(0.0, 1.0 - assisted), 4),
            "risky_window_density": round(max(0.0, assisted - 0.35), 4),
            "high_confidence_risky_window_count": 0,
            "max_risky_window_words": 0,
            "top_risky_windows": [],
            "confidence": "low",
        }
    return {}


def _rewrite_target_profile(report: dict | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    candidates = [
        report.get("rewrite_target_profile"),
        (((report.get("scan_intelligence") or {}).get("document") or {}).get("rewrite_target_profile")),
        ((report.get("scan_intelligence") or {}).get("rewrite_target_profile")),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _problem_inventory(report: dict | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    candidates = [
        report.get("problem_inventory"),
        (((report.get("scan_intelligence") or {}).get("document") or {}).get("problem_inventory")),
        ((report.get("scan_intelligence") or {}).get("problem_inventory")),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _footprint_risk(profile: dict[str, Any] | None) -> float:
    payload = profile if isinstance(profile, dict) else {}
    return round(
        _number(payload.get("fraction_ai")) * 100.0
        + _number(payload.get("fraction_ai_assisted")) * 45.0
        + _number(payload.get("risky_window_density")) * 35.0
        + min(_number(payload.get("max_risky_window_words")), 260.0) * 0.04
        + _number(payload.get("high_confidence_risky_window_count")) * 4.0,
        3,
    )


def _footprint_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_risk = _footprint_risk(before)
    after_risk = _footprint_risk(after)
    fraction_ai_drop = _number(before.get("fraction_ai")) - _number(after.get("fraction_ai"))
    assisted_drop = _number(before.get("fraction_ai_assisted")) - _number(after.get("fraction_ai_assisted"))
    risky_density_drop = _number(before.get("risky_window_density")) - _number(after.get("risky_window_density"))
    high_conf_drop = _number(before.get("high_confidence_risky_window_count")) - _number(after.get("high_confidence_risky_window_count"))
    risk_drop = before_risk - after_risk
    risk_not_worse = risk_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MAX_FOOTPRINT_BACKFIRE", 0.0)
    moved = bool(
        risk_not_worse
        and (
            risk_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_FOOTPRINT_RISK_DROP", 2.0)
            or fraction_ai_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_FRACTION_AI_DROP", 0.03)
            or assisted_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_ASSISTED_FRACTION_DROP", 0.04)
            or risky_density_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_RISKY_DENSITY_DROP", 0.04)
            or high_conf_drop >= 1.0
        )
    )
    return {
        "before_risk": before_risk,
        "after_risk": after_risk,
        "risk_drop": round(risk_drop, 3),
        "fraction_ai_drop": round(fraction_ai_drop, 4),
        "fraction_ai_assisted_drop": round(assisted_drop, 4),
        "risky_window_density_drop": round(risky_density_drop, 4),
        "high_confidence_risky_window_drop": round(high_conf_drop, 3),
        "risk_not_worse": bool(risk_not_worse),
        "moved": bool(moved),
    }


def _target_profile_risk(profile: dict[str, Any] | None) -> float:
    payload = profile if isinstance(profile, dict) else {}
    targets = payload.get("targets") if isinstance(payload.get("targets"), list) else []
    level_weights = {"high": 3.0, "medium": 2.0, "low": 1.0, "minimal": 0.25}
    risk = 0.0
    for target in targets:
        if not isinstance(target, dict):
            continue
        level = str(target.get("risk_level") or "minimal")
        risk += level_weights.get(level, 0.5)
        for driver in target.get("dominant_drivers") or []:
            if isinstance(driver, dict):
                risk += min(_number(driver.get("score")), 1.0)
    return round(risk, 3)


def _target_profile_movement(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_targets = before.get("targets") if isinstance(before.get("targets"), list) else []
    after_targets = after.get("targets") if isinstance(after.get("targets"), list) else []
    before_risk = _target_profile_risk(before)
    after_risk = _target_profile_risk(after)
    risk_drop = before_risk - after_risk
    count_drop = len(before_targets) - len(after_targets)
    available = bool(before_targets)
    moved = (
        not available
        or risk_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_TARGET_RISK_DROP", 1.0)
        or count_drop >= 1
    )
    return {
        "available": available,
        "before_risk": before_risk,
        "after_risk": after_risk,
        "risk_drop": round(risk_drop, 3),
        "before_target_count": len(before_targets),
        "after_target_count": len(after_targets),
        "target_count_drop": count_drop,
        "moved": bool(moved),
    }


def _ownership_summary_from_target_trace(target_trace: dict[str, Any] | None) -> dict[str, Any]:
    trace = target_trace if isinstance(target_trace, dict) else {}
    quality_rows: list[dict[str, Any]] = []
    for key in ("accepted_replacements", "target_replacements", "scanner_controlled_accepted"):
        rows = trace.get(key) if isinstance(trace.get(key), list) else []
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("candidate_quality"), dict):
                quality_rows.append(row["candidate_quality"])
    for row in trace.get("prompt_stage_trace") or []:
        if not isinstance(row, dict):
            continue
        for group in row.get("groups") or []:
            if not isinstance(group, dict):
                continue
            quality = group.get("accepted_candidate_quality")
            if isinstance(quality, dict):
                quality_rows.append(quality)

    ownership_elements: set[str] = set()
    ownership_score = 0.0
    ownership_change_count = 0
    for quality in quality_rows:
        ownership_score = max(ownership_score, _number(quality.get("ownership_score")))
        ownership_change_count += int(quality.get("ownership_change_count") or 0)
        ownership_elements.update(
            str(item)
            for item in quality.get("ownership_elements_supported") or []
            if str(item or "")
        )
    return {
        "ownership_score": round(ownership_score, 3),
        "ownership_change_count": ownership_change_count,
        "ownership_elements_supported": sorted(ownership_elements),
        "quality_rows": len(quality_rows),
    }


def _ownership_gate(*, proxy_result: Any, target_trace: dict[str, Any] | None, goal_result: Any) -> dict[str, Any]:
    proxy_payload = proxy_result.to_dict() if hasattr(proxy_result, "to_dict") else {}
    reasons = {
        str(reason)
        for reason in proxy_payload.get("reasons") or []
        if str(reason)
    }
    segment_blockers = {
        "segment_ai_fraction_high",
        "segment_ai_or_assisted_fraction_high",
        "segment_human_fraction_low",
        "segment_ai_window_too_large",
        "high_confidence_ai_window_remaining",
    }
    goal_payload = goal_result.to_dict() if hasattr(goal_result, "to_dict") else {}
    eligible = goal_payload.get("eligible_span_density_gate") if isinstance(goal_payload.get("eligible_span_density_gate"), dict) else {}
    active = bool(segment_blockers.intersection(reasons) or eligible.get("needs_author_context"))
    summary = _ownership_summary_from_target_trace(target_trace)
    passed = (
        not active
        or _number(summary.get("ownership_score")) > 0.0
        or int(summary.get("ownership_change_count") or 0) > 0
    )
    return {
        "active": active,
        "passed": bool(passed),
        "reason": "" if passed else "ownership_required_for_human_fraction_blocker",
        **summary,
    }


def _text_integrity(source_text: str, candidate_text: str) -> dict[str, Any]:
    source = str(source_text or "")
    candidate = str(candidate_text or "")

    def normalized_alpha_tokens(text: str) -> list[str]:
        tokens: list[str] = []
        current: list[str] = []
        for char in str(text or "").casefold():
            if char.isalpha():
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return tokens

    def near_source_token(token: str, source_tokens: set[str]) -> bool:
        if token in source_tokens:
            return True
        if len(token) < 4:
            return False
        return any(
            abs(len(source_token) - len(token)) <= 2
            and (source_token.startswith(token) or token.startswith(source_token))
            for source_token in source_tokens
            if len(source_token) >= 4
        )

    def merged_source_token_candidates(source_text: str, candidate_text: str) -> list[dict[str, Any]]:
        source_tokens = set(normalized_alpha_tokens(source_text))
        candidate_tokens = normalized_alpha_tokens(candidate_text)
        rows: list[dict[str, Any]] = []
        for token in candidate_tokens:
            if token in source_tokens or len(token) < 9:
                continue
            for split_at in range(3, len(token) - 2):
                left = token[:split_at]
                right = token[split_at:]
                if near_source_token(left, source_tokens) and near_source_token(right, source_tokens):
                    rows.append({"token": token, "split": [left, right]})
                    break
            if len(rows) >= 8:
                break
        return rows

    def metrics(text: str) -> dict[str, Any]:
        chars = list(text)
        char_count = max(1, len(chars))
        tokens = text.split()
        alpha_run = 0
        max_alpha_run = 0
        non_alnum_run = 0
        max_non_alnum_run = 0
        punctuation = 0
        non_ascii_punctuation = 0
        symbol = 0
        emoji_like = 0
        zero_width = 0
        control_or_unassigned = 0
        for char in chars:
            if char.isalpha():
                alpha_run += 1
                max_alpha_run = max(max_alpha_run, alpha_run)
            else:
                alpha_run = 0
            if char.isalnum() or char.isspace():
                non_alnum_run = 0
            else:
                non_alnum_run += 1
                max_non_alnum_run = max(max_non_alnum_run, non_alnum_run)
            category = unicodedata.category(char)
            if category.startswith("P"):
                punctuation += 1
                if ord(char) > 127:
                    non_ascii_punctuation += 1
            if category.startswith("S"):
                symbol += 1
            if category == "Cf":
                zero_width += 1
            if category.startswith("C") and category not in {"Cc", "Cf"}:
                control_or_unassigned += 1
            if ord(char) >= 0x1F000:
                emoji_like += 1
        long_tokens = [token for token in tokens if len(token) >= 24]
        return {
            "char_count": len(chars),
            "token_count": len(tokens),
            "space_ratio": sum(1 for char in chars if char.isspace()) / char_count,
            "punctuation_ratio": punctuation / char_count,
            "non_ascii_punctuation_ratio": non_ascii_punctuation / char_count,
            "symbol_ratio": symbol / char_count,
            "emoji_like_count": emoji_like,
            "zero_width_count": zero_width,
            "control_or_unassigned_count": control_or_unassigned,
            "max_alpha_run": max_alpha_run,
            "max_non_alnum_run": max_non_alnum_run,
            "long_token_ratio": len(long_tokens) / max(1, len(tokens)),
        }

    src = metrics(source)
    cand = metrics(candidate)
    merged_candidates = merged_source_token_candidates(source, candidate)
    failures: list[str] = []
    warnings: list[str] = []
    if candidate.strip() and cand["space_ratio"] < max(0.02, src["space_ratio"] * 0.55):
        failures.append("spacing_collapse")
    if cand["max_alpha_run"] >= max(36, src["max_alpha_run"] * 2):
        failures.append("merged_word_run")
    if cand["long_token_ratio"] > max(0.04, src["long_token_ratio"] + 0.035):
        failures.append("merged_long_tokens")
    if merged_candidates and (
        failures
        or any(len(str(row.get("token") or "")) >= 18 for row in merged_candidates if isinstance(row, dict))
    ):
        failures.append("merged_source_token")
    elif merged_candidates:
        warnings.append("merged_source_token")
    if cand["non_ascii_punctuation_ratio"] > max(0.025, src["non_ascii_punctuation_ratio"] + 0.02):
        failures.append("punctuation_script_shift")
    if cand["punctuation_ratio"] > max(0.18, src["punctuation_ratio"] + 0.10):
        failures.append("punctuation_density_shift")
    if cand["symbol_ratio"] > max(0.03, src["symbol_ratio"] + 0.02):
        failures.append("unicode_symbol_burst")
    if cand["emoji_like_count"] > src["emoji_like_count"]:
        failures.append("emoji_or_decorative_symbol_injection")
    if cand["zero_width_count"] > src["zero_width_count"]:
        failures.append("zero_width_character_injection")
    if cand["control_or_unassigned_count"] > src["control_or_unassigned_count"]:
        failures.append("control_or_unassigned_character_injection")
    if cand["max_non_alnum_run"] > max(12, src["max_non_alnum_run"] + 8):
        failures.append("non_language_symbol_run")
    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "source": src,
        "candidate": cand,
        "merged_source_token_candidates": merged_candidates,
    }


def _validity_status(*, validation_passed: bool, compression_ok: bool, semantic_safe: bool, integrity_passed: bool) -> str:
    if validation_passed and compression_ok and semantic_safe and integrity_passed:
        return "valid"
    return "invalid"


def _candidate_outcome(*, validity_status: str, detector_moved: bool, integrity_passed: bool) -> str:
    if not integrity_passed:
        return "corrupted_output"
    if validity_status == "valid" and detector_moved:
        return "valid_detector_improved"
    if validity_status != "valid" and detector_moved:
        return "invalid_detector_improved"
    if validity_status == "valid":
        return "valid_no_detector_movement"
    return "invalid_no_detector_movement"


def _generation_failure_outcome(*, text: str, error: str | None) -> str | None:
    if text:
        return None
    if error:
        return "generation_failed_provider"
    return "generation_failed_empty_output"


def _semantic_safe_for_v3(semantic_result: Any) -> bool:
    if bool(getattr(semantic_result, "accepted", False)):
        return True
    similarity = _number(getattr(semantic_result, "similarity", 0.0), 0.0)
    if similarity < 0.985:
        return False
    reasons = [str(reason) for reason in (getattr(semantic_result, "reasons", []) or [])]
    if not reasons:
        return False
    punctuation_only_reasons = [
        reason for reason in reasons
        if reason.startswith("citation_lost:") or reason.startswith("quote_lost:")
    ]
    return len(punctuation_only_reasons) == len(reasons)


def _authorship_window_profile(report: dict | None) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    direct = report.get("authorship_window_profile")
    if isinstance(direct, dict) and direct:
        return direct
    scan_intel = report.get("scan_intelligence") if isinstance(report.get("scan_intelligence"), dict) else {}
    profile = scan_intel.get("authorship_window_profile")
    if isinstance(profile, dict) and profile:
        return profile
    document = scan_intel.get("document") if isinstance(scan_intel.get("document"), dict) else {}
    profile = document.get("authorship_window_profile")
    return profile if isinstance(profile, dict) else {}


def _logical_unit_count(original_text: str, report: dict | None) -> int:
    return len(_logical_source_units(original_text, report))


def _logical_source_units(original_text: str, report: dict | None) -> list[dict[str, Any]]:
    structural_units = document_units(original_text)
    structural_payload = [unit.to_dict() for unit in structural_units]
    profile = _authorship_window_profile(report)
    windows = profile.get("windows") if isinstance(profile.get("windows"), list) else []
    if len(structural_units) != 1 or len(windows) <= 1:
        return structural_payload

    text = str(original_text or "")
    sorted_windows = sorted(
        [window for window in windows if isinstance(window, dict)],
        key=lambda window: int(window.get("start_index") or 0),
    )
    units: list[dict[str, Any]] = []
    previous_end = 0
    for index, window in enumerate(sorted_windows, start=1):
        raw_end = int(window.get("end_index") or previous_end)
        end = max(previous_end, min(len(text), raw_end))
        if index == len(sorted_windows):
            end = len(text)
        unit_text = text[previous_end:end].strip()
        previous_end = end
        if not unit_text:
            continue
        units.append({
            "unit_id": f"u{len(units) + 1}",
            "text": unit_text,
            "word_count": word_count(unit_text),
            "is_heading": False,
            "source_window_id": window.get("window_id"),
            "source_paragraph_id": window.get("paragraph_id"),
        })
    return units or structural_payload


def _single_shot_word_limit() -> int:
    try:
        return max(300, int(os.environ.get("DRAFTPROOF_REWRITE_V3_SINGLE_SHOT_WORD_LIMIT", "1600") or 1600))
    except (TypeError, ValueError):
        return 1600


def _chunk_word_limit() -> int:
    try:
        return max(250, int(os.environ.get("DRAFTPROOF_REWRITE_V3_CHUNK_WORD_LIMIT", "900") or 900))
    except (TypeError, ValueError):
        return 900


def _llm_timeout_seconds() -> int:
    try:
        return max(20, int(os.environ.get("DRAFTPROOF_REWRITE_V3_LLM_TIMEOUT_SECONDS", "160") or 160))
    except (TypeError, ValueError):
        return 160


def _max_tokens_for_words(words: int) -> int:
    floor = _int_env("DRAFTPROOF_REWRITE_V3_MIN_COMPLETION_TOKENS", 5000)
    cap = _int_env("DRAFTPROOF_REWRITE_V3_MAX_COMPLETION_TOKENS", 12000)
    floor = max(1200, floor)
    cap = max(floor, cap)
    return max(floor, min(cap, int(words * 2.4) + 900))


def _reject_length_limited_response(response: Any, *, stage: str) -> str:
    finish_reason = str(getattr(response, "finish_reason", "") or "").strip().lower()
    if finish_reason == "length":
        logger.warning("Rejected V3 LLM response truncated by max_tokens at stage=%s", stage)
        return ""
    return str(getattr(response, "content", "") or "")


def _reject_length_limited_raw_response(response: Any, *, stage: str) -> str:
    finish_reason = str(getattr(response, "finish_reason", "") or "").strip().lower()
    if finish_reason == "length":
        logger.warning("Rejected V3 LLM response truncated by max_tokens at stage=%s", stage)
        return ""
    raw_content = getattr(response, "raw_content", "")
    if raw_content:
        return str(raw_content)
    return str(getattr(response, "content", "") or "")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _compression_accepted(compression: dict[str, Any]) -> bool:
    if compression.get("in_band"):
        return True
    policy = compression.get("policy") if isinstance(compression.get("policy"), dict) else {}
    source_words = int(compression.get("source_words") or 0)
    candidate_words = int(compression.get("candidate_words") or 0)
    tolerance = max(0, int(round(source_words * max(0.0, _float_env("DRAFTPROOF_REWRITE_V3_COMPRESSION_TOLERANCE_RATIO", 0.02)))))
    if compression.get("status") == "above_ceiling":
        return candidate_words <= int(policy.get("max_words") or 0) + tolerance
    if compression.get("status") == "below_floor":
        return candidate_words >= int(policy.get("min_words") or 0) - tolerance
    return False


def _family_for_route(content_mode: str, contract_anchor_count: int) -> str:
    if content_mode == "academic_cited_text" or contract_anchor_count >= 3:
        return "cited_practice_voice"
    if os.environ.get("DRAFTPROOF_REWRITE_V3_CLEAN_TEXTURE_ENABLED", "1").lower() not in {"0", "false", "no", "off"}:
        if content_mode in {"broad_explanatory_essay", "generic_expository"}:
            return "clean_texture_boundary"
    return "document_rhythm"


def _clean_texture_chat_kwargs() -> dict[str, float]:
    return {
        "temperature": _float_env("DRAFTPROOF_REWRITE_V3_CLEAN_TEXTURE_TEMPERATURE", 0.72),
        "top_p": _float_env("DRAFTPROOF_REWRITE_V3_CLEAN_TEXTURE_TOP_P", 0.93),
        "presence_penalty": _float_env("DRAFTPROOF_REWRITE_V3_CLEAN_TEXTURE_PRESENCE_PENALTY", 0.18),
        "frequency_penalty": _float_env("DRAFTPROOF_REWRITE_V3_CLEAN_TEXTURE_FREQUENCY_PENALTY", 0.30),
        "repetition_penalty": _float_env("DRAFTPROOF_REWRITE_V3_CLEAN_TEXTURE_REPETITION_PENALTY", 1.05),
    }


def _should_use_chunked_generation(
    *,
    source_words: int,
    scan_contract: ScanContract,
    v3_route: Any,
    exact_anchor_count: int,
) -> bool:
    if source_words > _single_shot_word_limit():
        return True
    if (
        scan_contract.anchor_preservation_pressure >= 0.55
        and source_words > max(650, int(_single_shot_word_limit() * 0.75))
        and exact_anchor_count >= 8
    ):
        return True
    return (
        v3_route.primary_class == RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY
        and scan_contract.evidence_anchor_score >= 0.5
        and source_words > max(650, int(_single_shot_word_limit() * 0.75))
    )


def _should_force_unit_chunks(*, scan_contract: ScanContract, v3_route: Any, exact_anchor_count: int) -> bool:
    if scan_contract.word_count <= _single_shot_word_limit():
        return False
    if scan_contract.anchor_preservation_pressure >= 0.55 and exact_anchor_count >= 8:
        return True
    protected_classes = {
        RewriteRiskClass.CITED_ACADEMIC,
        RewriteRiskClass.TECHNICAL_STRUCTURED,
        RewriteRiskClass.REGULATED_POLICY,
    }
    if v3_route.primary_class in protected_classes:
        return True
    return (
        v3_route.primary_class == RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY
        and scan_contract.evidence_anchor_score >= 0.5
    )


def _restore_exact_quote_anchors(text: str, contract: Any) -> str:
    repaired = str(text or "")
    quote_chars = {'"', "'", "“", "”", "‘", "’"}
    for anchor in getattr(contract, "anchors", []) or []:
        anchor_text = str(getattr(anchor, "text", "") or "")
        if not anchor_text or anchor_text in repaired:
            continue
        kind = str(getattr(anchor, "kind", "") or "")
        severity = getattr(getattr(anchor, "severity", None), "value", "")
        if kind != "direct_quote" or severity != "hard_exact":
            continue
        inner = anchor_text.strip().strip("\"'“”‘’").strip()
        if not inner:
            continue
        start = repaired.find(inner)
        if start < 0:
            continue
        left = start - 1 if start > 0 and repaired[start - 1] in quote_chars else start
        right = start + len(inner)
        while right < len(repaired) and repaired[right] in ".,;:!?":
            right += 1
        if right < len(repaired) and repaired[right] in quote_chars:
            right += 1
        repaired = repaired[:left] + anchor_text + repaired[right:]
    return repaired


def _v3_content_mode(scan_contract: ScanContract, v3_route: Any) -> tuple[str, dict[str, Any]]:
    scan_mode = str(scan_contract.content_mode or "").strip()
    confidence = float(scan_contract.content_mode_confidence or 0.0)
    if scan_mode and scan_mode != "unknown" and confidence > 0:
        return scan_mode, {
            "content_mode": scan_mode,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "reasons": ["scan_contract_content_mode", *list(v3_route.reasons)],
            "mode_scores": list(scan_contract.mode_scores),
            "router_source": "rewrite_v3_scan_contract",
        }

    fallback_by_class = {
        RewriteRiskClass.BROAD_PROSE: "broad_explanatory_essay",
        RewriteRiskClass.CITED_ACADEMIC: "academic_cited_text",
        RewriteRiskClass.TECHNICAL_STRUCTURED: "technical_content",
        RewriteRiskClass.REGULATED_POLICY: "regulated_policy_content",
        RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY: "quote_heavy",
        RewriteRiskClass.PERSONAL_REFLECTIVE: "personal_reflection",
        RewriteRiskClass.CREATIVE_MARKETING: "creative_marketing",
        RewriteRiskClass.SHORT_OR_SPARSE: "short_text",
    }
    content_mode = fallback_by_class.get(v3_route.primary_class, "generic_expository")
    return content_mode, {
        "content_mode": content_mode,
        "confidence": v3_route.confidence,
        "reasons": ["derived_from_v3_route", *list(v3_route.reasons)],
        "mode_scores": [
            {"content_mode": content_mode, "score": v3_route.confidence, "source": "rewrite_v3_route"}
        ],
        "router_source": "rewrite_v3_route_fallback",
    }


def _gateway(api_key: str | None, model: str | None, base_url: str | None, *, max_tokens: int) -> LLMGateway:
    def active_model() -> str:
        for value in (
            os.environ.get("DRAFTPROOF_REWRITE_MODEL_LOCK"),
            os.environ.get("DRAFTPROOF_GENERATOR_MODEL"),
            model,
            os.environ.get("LLM_MODEL"),
        ):
            candidate = str(value or "").strip()
            if candidate and candidate.lower() not in {"0", "false", "no", "off"}:
                return candidate
        return "deepseek/deepseek-v4-flash"

    return LLMGateway(LLMConfig(
        api_key=api_key,
        model=active_model(),
        base_url=base_url or os.environ.get("LLM_BASE_URL", ""),
        max_tokens=max_tokens,
        temperature=float(os.environ.get("DRAFTPROOF_REWRITE_V3_TEMPERATURE", "0.82") or 0.82),
        top_p=float(os.environ.get("DRAFTPROOF_REWRITE_V3_TOP_P", "0.93") or 0.93),
        presence_penalty=float(os.environ.get("DRAFTPROOF_REWRITE_V3_PRESENCE_PENALTY", "0.42") or 0.42),
        frequency_penalty=float(os.environ.get("DRAFTPROOF_REWRITE_V3_FREQUENCY_PENALTY", "0.40") or 0.40),
        repetition_penalty=float(os.environ.get("DRAFTPROOF_REWRITE_V3_REPETITION_PENALTY", "1.07") or 1.07),
        timeout=_llm_timeout_seconds(),
        max_retries=1,
    ))


def _generate_single_candidate(
    *,
    original_text: str,
    original_report: dict[str, Any] | None = None,
    family: str,
    contract: Any,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    rewrite_target_profile: dict[str, Any] | None = None,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    central_judgment_plan: dict[str, Any] | None = None,
) -> str:
    examples = examples_for_family(family)
    if family == "cited_practice_voice":
        prompt = build_cited_practice_voice_prompt(
            original_text=original_text,
            contract=contract,
            compression_policy=compression_policy,
            style_examples=examples,
            rewrite_target_profile=rewrite_target_profile,
            predictability_briefs=predictability_briefs,
            central_judgment_plan=central_judgment_plan,
        )
    elif family == "clean_texture_boundary":
        prompt = build_clean_texture_boundary_prompt(
            original_text=original_text,
            scan_report=original_report,
            style_examples=examples,
            rewrite_target_profile=rewrite_target_profile,
            predictability_briefs=predictability_briefs,
            central_judgment_plan=central_judgment_plan,
        )
    else:
        prompt = build_document_rhythm_prompt(
            original_text=original_text,
            compression_policy=compression_policy,
            style_examples=examples,
            rewrite_target_profile=rewrite_target_profile,
            predictability_briefs=predictability_briefs,
            central_judgment_plan=central_judgment_plan,
        )
    token_words = word_count(original_text) if family == "clean_texture_boundary" else compression_policy.max_words
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(token_words))
    kwargs = _clean_texture_chat_kwargs() if family == "clean_texture_boundary" else {}
    response = gateway.chat(
        prompt,
        system="Return only the rewritten document as plain text.",
        **kwargs,
    )
    return clean_v3_candidate_output(_reject_length_limited_response(response, stage=f"{family}:initial"))


def _generate_recovery_candidate(
    *,
    original_text: str,
    failed_candidate: str,
    family: str,
    proxy_feedback: dict[str, Any],
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_recovery_revision_prompt(
        original_text=original_text,
        failed_candidate=failed_candidate,
        strategy_family=family,
        proxy_feedback=proxy_feedback,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
    )
    token_words = max(compression_policy.max_words, word_count(original_text) + 420)
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(token_words))
    response = gateway.chat(prompt, system="Return only the rewritten document as plain text.")
    return clean_v3_candidate_output(_reject_length_limited_response(response, stage="recovery_revision"))


def _generate_contract_repair_candidate(
    *,
    original_text: str,
    failed_candidate: str,
    family: str,
    candidate_trace: dict[str, Any],
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_contract_repair_prompt(
        original_text=original_text,
        failed_candidate=failed_candidate,
        strategy_family=family,
        candidate_trace=candidate_trace,
        compression_policy=compression_policy,
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the repaired rewritten document as plain text.").content)


def _generate_boundary_candidate(
    *,
    original_text: str,
    failed_candidates: list[str],
    family: str,
    proxy_feedback: list[dict[str, Any]],
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_boundary_adapter_prompt(
        original_text=original_text,
        failed_candidates=failed_candidates,
        strategy_family=family,
        proxy_feedback=proxy_feedback,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the rewritten document as plain text.").content)


def _generate_contrast_boundary_candidate(
    *,
    original_text: str,
    failed_candidate: str,
    family: str,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_contrast_boundary_prompt(
        original_text=original_text,
        failed_candidate=failed_candidate,
        family=family,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    raw = gateway.chat(prompt, system="Return only valid JSON with rewritten_document and notes.").content
    return clean_v3_candidate_output(extract_contrast_boundary_output(raw))


def _generate_plain_reasoning_candidate(
    *,
    original_text: str,
    failed_candidates: list[str],
    family: str,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    rewrite_target_profile: dict[str, Any] | None = None,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    central_judgment_plan: dict[str, Any] | None = None,
) -> str:
    prompt = build_plain_reasoning_broad_prose_prompt(
        original_text=original_text,
        failed_candidates=failed_candidates,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
        rewrite_target_profile=rewrite_target_profile,
        predictability_briefs=predictability_briefs,
        central_judgment_plan=central_judgment_plan,
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the rewritten document as plain text.").content)


def _max_authorship_window_repair_targets() -> int:
    try:
        return max(1, min(4, int(os.environ.get("DRAFTPROOF_REWRITE_V3_WINDOW_REPAIR_TARGETS", "2") or 2)))
    except (TypeError, ValueError):
        return 2


def _max_target_executor_groups() -> int:
    try:
        explicit = os.environ.get("DRAFTPROOF_REWRITE_V3_MAX_TARGET_GROUPS")
        value = explicit if explicit is not None else os.environ.get("DRAFTPROOF_REWRITE_V3_TARGET_GROUPS", "12")
        return max(1, min(24, int(value or 12)))
    except (TypeError, ValueError):
        return 12


def _target_executor_batch_size() -> int:
    try:
        return max(1, min(6, int(os.environ.get("DRAFTPROOF_REWRITE_V3_TARGET_BATCH_SIZE", "4") or 4)))
    except (TypeError, ValueError):
        return 4


def _max_assisted_footprint_groups() -> int:
    try:
        return max(1, min(18, int(os.environ.get("DRAFTPROOF_REWRITE_V3_ASSISTED_GROUPS", "10") or 10)))
    except (TypeError, ValueError):
        return 10


def _assisted_footprint_batch_size() -> int:
    try:
        return max(1, min(5, int(os.environ.get("DRAFTPROOF_REWRITE_V3_ASSISTED_BATCH_SIZE", "3") or 3)))
    except (TypeError, ValueError):
        return 3


def _assisted_footprint_executor_available(scan_contract: ScanContract) -> bool:
    if os.environ.get("DRAFTPROOF_REWRITE_V3_ASSISTED_LAYER_ENABLED", "0").lower() not in {"1", "true", "yes", "on"}:
        return False
    broad_assisted = (
        scan_contract.footprint_fraction_ai + scan_contract.footprint_fraction_ai_assisted
    ) >= _float_env("DRAFTPROOF_REWRITE_V3_ASSISTED_LAYER_MIN_FRACTION", 0.55)
    return bool(broad_assisted and scan_contract.risky_window_count + scan_contract.high_confidence_risky_window_count >= 0)


def _scanner_controlled_config() -> ScannerControlledConfig:
    def int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
        try:
            value = int(os.environ.get(name, str(default)) or default)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    return ScannerControlledConfig(
        max_rounds=int_env("DRAFTPROOF_REWRITE_V3_SCANNER_LOOP_ROUNDS", 2, minimum=1, maximum=3),
        groups_per_round=int_env("DRAFTPROOF_REWRITE_V3_SCANNER_LOOP_GROUPS", 4, minimum=1, maximum=8),
        variants_per_group=int_env("DRAFTPROOF_REWRITE_V3_SCANNER_LOOP_VARIANTS", 3, minimum=1, maximum=4),
        min_accept_delta=_float_env("DRAFTPROOF_REWRITE_V3_SCANNER_LOOP_MIN_DELTA", 0.2),
    )


def _scanner_controlled_executor_available(scan_contract: ScanContract) -> bool:
    if os.environ.get("DRAFTPROOF_REWRITE_V3_SCANNER_CONTROLLED_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        return False
    return _target_executor_available(scan_contract)


def _scanner_controlled_should_run_first(scan_contract: ScanContract) -> bool:
    if os.environ.get("DRAFTPROOF_REWRITE_V3_SCANNER_CONTROLLED_FIRST_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        return False
    if not _scanner_controlled_executor_available(scan_contract):
        return False
    topk_score = scan_contract.topk_score
    if isinstance(topk_score, (int, float)) and topk_score >= _float_env("DRAFTPROOF_REWRITE_V3_SCANNER_FIRST_TOPK_MIN", 80.0):
        return True
    drivers = scan_contract.target_driver_summary or {}
    unsafe_driver = int(drivers.get("unsafe_word_share") or 0)
    predictability_driver = int(drivers.get("predictability_score") or 0)
    if unsafe_driver >= _int_env("DRAFTPROOF_REWRITE_V3_SCANNER_FIRST_UNSAFE_DRIVER_MIN", 5):
        return True
    if predictability_driver >= _int_env("DRAFTPROOF_REWRITE_V3_SCANNER_FIRST_PREDICTABILITY_DRIVER_MIN", 7):
        return True
    unsafe_score_min = _float_env("DRAFTPROOF_REWRITE_V3_SCANNER_FIRST_UNSAFE_SCORE_MIN", 0.55)
    predictability_score_min = _float_env("DRAFTPROOF_REWRITE_V3_SCANNER_FIRST_PREDICTABILITY_SCORE_MIN", 0.60)
    for target in scan_contract.rewrite_targets:
        if not isinstance(target, dict):
            continue
        for driver in target.get("dominant_drivers") or []:
            if not isinstance(driver, dict):
                continue
            key = str(driver.get("key") or "")
            score = driver.get("score")
            if not isinstance(score, (int, float)):
                continue
            if key == "unsafe_word_share" and float(score) >= unsafe_score_min:
                return True
            if key == "predictability_score" and float(score) >= predictability_score_min:
                return True
    return False


def _unbounded_recovery_enabled() -> bool:
    return os.environ.get("DRAFTPROOF_REWRITE_V3_ALLOW_UNBOUNDED_RECOVERY", "0").lower() in {"1", "true", "yes", "on"}


def _target_executor_available(scan_contract: ScanContract) -> bool:
    if not scan_contract.rewrite_targets:
        return False
    operations = set(scan_contract.target_operation_mix or {})
    if operations.intersection(SUPPORTED_TARGET_OPERATIONS):
        return True
    return any(
        str(target.get("recommended_operation") or "") in SUPPORTED_TARGET_OPERATIONS
        for target in scan_contract.rewrite_targets
        if isinstance(target, dict)
    )


def _prune_bridge_available(scan_contract: ScanContract) -> bool:
    for group in scan_contract.problem_groups:
        if not isinstance(group, dict):
            continue
        allowed = {str(item) for item in group.get("allowed_operations") or []}
        if "unit_preserving_prune_bridge" in allowed:
            return True
    return False


PROBLEM_SCANNER_CONTROLLED_STRATEGIES = {
    "authorship_window_repair",
    "citation_anchor_guard",
    "citation_preserving_window_repair",
    "paragraph_surgery",
    "protected_section_rewrite",
}


PROBLEM_TARGET_EXECUTOR_STRATEGIES = {
    "paragraph_preserving_broad_reconstruction",
}


def _uses_problem_scanner_controlled_strategy(strategy_id: str) -> bool:
    return strategy_id in PROBLEM_SCANNER_CONTROLLED_STRATEGIES


def _uses_problem_target_executor_strategy(strategy_id: str) -> bool:
    return strategy_id in PROBLEM_TARGET_EXECUTOR_STRATEGIES


def _unapplied_target_group_ids(target_trace: dict[str, Any] | None) -> list[str]:
    trace = target_trace if isinstance(target_trace, dict) else {}
    rows = trace.get("target_apply_status") if isinstance(trace.get("target_apply_status"), list) else []
    return [
        str(row.get("group_id") or "")
        for row in rows
        if isinstance(row, dict)
        and str(row.get("group_id") or "")
        and not bool(row.get("applied"))
    ]


def _topk_effect_failure_rows(target_trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    trace = target_trace if isinstance(target_trace, dict) else {}
    rows: list[dict[str, Any]] = []
    stage_trace = trace.get("prompt_stage_trace") if isinstance(trace.get("prompt_stage_trace"), list) else []
    for stage in stage_trace:
        if not isinstance(stage, dict):
            continue
        diagnostics = stage.get("parse_diagnostics") if isinstance(stage.get("parse_diagnostics"), dict) else {}
        effect_status = diagnostics.get("effect_status") if isinstance(diagnostics.get("effect_status"), list) else []
        for status in effect_status:
            if not isinstance(status, dict):
                continue
            failures = [str(item) for item in status.get("failures") or [] if str(item or "")]
            if not failures:
                continue
            rows.append({
                "group_id": str(status.get("group_id") or ""),
                "failures": list(dict.fromkeys(failures)),
                "required_modified_spans": status.get("required_modified_spans"),
                "actual_predictable_spans_modified_count": status.get("actual_predictable_spans_modified_count"),
                "actual_modified_span_ids": status.get("actual_modified_span_ids") if isinstance(status.get("actual_modified_span_ids"), list) else [],
            })
    return rows


def _topk_effect_failures(target_trace: dict[str, Any] | None) -> list[str]:
    failures: list[str] = []
    for row in _topk_effect_failure_rows(target_trace):
        failures.extend(str(item) for item in row.get("failures") or [] if str(item or ""))
    return list(dict.fromkeys(failures))


def _annotate_target_execution(
    trace: dict[str, Any],
    *,
    executed_strategy: str,
    executor_engine: str,
) -> dict[str, Any]:
    annotated = dict(trace or {})
    annotated["executed_strategy"] = executed_strategy
    if annotated.get("executor_engine"):
        annotated["dispatcher_engine"] = executor_engine
    else:
        annotated["executor_engine"] = executor_engine
    return annotated


def _strategy_plan_prefers_target_executor(strategy_plan: Any, scan_contract: ScanContract) -> bool:
    if not _target_executor_available(scan_contract):
        return False
    target_step_ids = {
        "protected_section_rewrite",
        "citation_anchor_guard",
        "citation_preserving_window_repair",
        "paragraph_surgery",
        "authorship_window_repair",
        "paragraph_preserving_broad_reconstruction",
    }
    for step in getattr(strategy_plan, "steps", ()) or ():
        step_id = str(getattr(step, "strategy_id", "") or "")
        if step_id == "portfolio_selection":
            continue
        if step_id in target_step_ids:
            return True
        if step_id in {"clean_texture_boundary", "document_rhythm", "plain_reasoning_broad_prose", "cited_practice_voice"}:
            return False
    return True


def _first_strategy_step_id(strategy_plan: Any) -> str:
    for step in getattr(strategy_plan, "steps", ()) or ():
        step_id = str(getattr(step, "strategy_id", "") or "")
        if step_id and step_id != "portfolio_selection":
            return step_id
    return ""


def _next_planned_problem_strategy(
    *,
    strategy_plan: Any,
    tried_strategy_ids: set[str],
    latest_trace: dict[str, Any],
) -> str:
    if not isinstance(latest_trace, dict):
        return ""
    if latest_trace.get("target_gate_passed", True) and not latest_trace.get("unapplied_target_group_ids"):
        return ""
    eligible = PROBLEM_SCANNER_CONTROLLED_STRATEGIES | PROBLEM_TARGET_EXECUTOR_STRATEGIES
    for step in getattr(strategy_plan, "steps", ()) or ():
        step_id = str(getattr(step, "strategy_id", "") or "")
        if not step_id or step_id in tried_strategy_ids or step_id not in eligible:
            continue
        return step_id
    return ""


def _missing_protected_anchors(text: str, group: Any) -> list[str]:
    if hasattr(group, "protected_anchors"):
        return missing_required_protected_anchors(text, group)
    return []


def _merge_replacements(
    base_replacements: list[dict[str, str]],
    patch_replacements: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged = {
        str(row.get("group_id") or ""): str(row.get("replacement_text") or "").strip()
        for row in base_replacements
        if str(row.get("group_id") or "") and str(row.get("replacement_text") or "").strip()
    }
    for row in patch_replacements:
        group_id = str(row.get("group_id") or "")
        replacement = str(row.get("replacement_text") or "").strip()
        if group_id and replacement:
            merged[group_id] = replacement
    return [
        {"group_id": group_id, "replacement_text": replacement}
        for group_id, replacement in merged.items()
    ]


def _generate_paragraph_portfolio_candidate(
    *,
    original_text: str,
    scan_contract: ScanContract,
    content_mode: str,
    family: str,
    target_groups: list[Any],
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> tuple[str, dict[str, Any]]:
    return _run_paragraph_portfolio_candidate(
        original_text=original_text,
        scan_contract=scan_contract,
        content_mode=content_mode,
        family=family,
        target_groups=target_groups,
        gateway_factory=lambda max_tokens: _gateway(api_key, model, base_url, max_tokens=max_tokens),
        token_budget=_max_tokens_for_words,
        config=paragraph_portfolio_config(fallback_batch_size=_target_executor_batch_size()),
    )


def _generate_paragraph_ownership_candidate(
    *,
    original_text: str,
    scan_contract: ScanContract,
    content_mode: str,
    family: str,
    target_groups: list[Any],
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> tuple[str, dict[str, Any]]:
    return _run_paragraph_ownership_candidate(
        original_text=original_text,
        scan_contract=scan_contract,
        content_mode=content_mode,
        family=family,
        target_groups=target_groups,
        gateway_factory=lambda max_tokens: _gateway(api_key, model, base_url, max_tokens=max_tokens),
        token_budget=_max_tokens_for_words,
        config=paragraph_portfolio_config(fallback_batch_size=_target_executor_batch_size()),
    )


def _generate_target_executor_candidate(
    *,
    original_text: str,
    scan_contract: ScanContract,
    content_mode: str,
    family: str,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> tuple[str, dict[str, Any]]:
    groups = group_rewrite_targets(
        original_text=original_text,
        rewrite_target_profile=scan_contract.rewrite_target_profile,
        max_groups=_max_target_executor_groups(),
    )
    if not groups:
        return "", target_execution_trace(
            attempted=True,
            target_groups=[],
            error="no_supported_target_groups",
        )
    if family == "paragraph_preserving_broad_reconstruction" or any(
        group.operation == "paragraph_preserving_broad_reconstruction"
        for group in groups
    ):
        return _generate_paragraph_portfolio_candidate(
            original_text=original_text,
            scan_contract=scan_contract,
            content_mode=content_mode,
            family=family,
            target_groups=groups,
            api_key=api_key,
            model=model,
            base_url=base_url,
        )
    replacements: list[dict[str, str]] = []
    batch_trace: list[dict[str, Any]] = []
    batch_errors: list[str] = []
    for batch_index, batch in enumerate(batch_target_groups(groups, batch_size=_target_executor_batch_size()), start=1):
        prompt = build_target_executor_prompt(
            target_groups=batch,
            content_mode=content_mode,
            strategy_family=family,
            predictability_briefs=scan_contract.predictability_briefs,
        )
        target_words = sum(int(group.word_count_guide.get("preferred_words") or word_count(group.source_text)) for group in batch)
        gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(max(220, target_words + 420)))
        batch_error = None
        try:
            raw = gateway.chat(
                prompt,
                system="Return only valid JSON with a replacements array.",
                response_format={"type": "json_object"},
            ).content
            batch_replacements = parse_target_replacements(raw)
        except Exception as exc:
            batch_error = str(exc)
            batch_errors.append(batch_error)
            batch_replacements = []
        replacements.extend(batch_replacements)
        batch_trace.append({
            "batch_index": batch_index,
            "group_ids": [group.group_id for group in batch],
            "requested_groups": len(batch),
            "replacement_count": len(batch_replacements),
            "error": batch_error,
        })
    if not replacements:
        return "", target_execution_trace(
            attempted=True,
            target_groups=groups,
            replacements=[],
            batches=batch_trace,
            error=batch_errors[0] if batch_errors else "generation_failed_empty_output",
        )
    text, apply_status = apply_target_replacements(
        original_text=original_text,
        target_groups=groups,
        replacements=replacements,
    )
    trace = target_execution_trace(
        attempted=True,
        target_groups=groups,
        replacements=replacements,
        apply_status=apply_status,
        batches=batch_trace,
        error="; ".join(batch_errors) if batch_errors else None,
    )
    if text.strip() == original_text.strip():
        trace = {**trace, "error": "no_target_replacement_applied"}
    return clean_v3_candidate_output(text), trace


def _generate_prune_bridge_candidate(
    *,
    original_text: str,
    scan_contract: ScanContract,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> tuple[str, dict[str, Any]]:
    base_groups = group_rewrite_targets(
        original_text=original_text,
        rewrite_target_profile=scan_contract.rewrite_target_profile,
        max_groups=max(_max_target_executor_groups(), 8),
    )
    groups = filter_prune_bridge_groups(
        target_groups=base_groups,
        problem_inventory=scan_contract.problem_inventory,
    )
    if not groups:
        return "", target_execution_trace(
            attempted=True,
            target_groups=[],
            error="no_prune_bridge_problem_groups",
        )
    prompt = build_prune_bridge_prompt(
        target_groups=groups[:_max_target_executor_groups()],
        predictability_briefs=scan_contract.predictability_briefs,
    )
    target_words = sum(int(group.word_count_guide.get("preferred_words") or word_count(group.source_text)) for group in groups)
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(max(180, target_words + 360)))
    try:
        raw = gateway.chat(
            prompt,
            system="Return only valid JSON with a replacements array.",
            response_format={"type": "json_object"},
        ).content
        replacements = parse_prune_bridge_replacements(raw)
    except Exception as exc:
        return "", target_execution_trace(
            attempted=True,
            target_groups=groups,
            error=str(exc),
        )
    if not replacements:
        return "", target_execution_trace(
            attempted=True,
            target_groups=groups,
            replacements=[],
            error="generation_failed_empty_output",
        )
    text, trace = apply_prune_bridge_replacements(
        original_text=original_text,
        target_groups=groups,
        replacements=replacements,
    )
    return clean_v3_candidate_output(text), trace


def _generate_scanner_controlled_candidate(
    *,
    original_text: str,
    original_report: dict[str, Any],
    scan_contract: ScanContract,
    content_mode: str,
    family: str,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    strategy_id: str | None = None,
    ownership_repair_mode: bool = False,
) -> tuple[str, dict[str, Any]]:
    config = _scanner_controlled_config()
    current_text = str(original_text or "")
    current_report = original_report
    original_goal = evaluate_rewrite_goal(
        original_text=original_text,
        candidate_text=original_text,
        original_report=original_report,
        candidate_report=original_report,
    ).to_dict()
    current_goal = original_goal
    current_metrics = scanner_controlled_metrics(
        report=current_report,
        goal=current_goal,
        footprint_risk=_footprint_risk(_ai_footprint_profile(current_report)),
        ai_score=_badge_ai(current_report),
        topk_score=_topk(current_report),
    )
    accepted: list[dict[str, Any]] = []
    accepted_unit_ids: set[str] = set()
    apply_statuses: list[dict[str, Any]] = []
    round_trace: list[dict[str, Any]] = []
    last_groups: list[Any] = []
    errors: list[str] = []

    if ownership_repair_mode:
        current_contract = build_scan_contract(current_report, current_text)
        ownership_groups = group_rewrite_targets(
            original_text=current_text,
            rewrite_target_profile=current_contract.rewrite_target_profile,
            max_groups=max(_max_target_executor_groups(), config.groups_per_round * 2),
        )
        if strategy_id:
            allowed_target_ids: set[str] = set()
            for problem_group in current_contract.problem_groups or []:
                if not isinstance(problem_group, dict):
                    continue
                allowed = {str(item) for item in problem_group.get("allowed_operations") or []}
                if strategy_id not in allowed:
                    continue
                allowed_target_ids.update(
                    str(target_id)
                    for target_id in problem_group.get("target_ids") or []
                    if str(target_id)
                )
            if allowed_target_ids:
                ownership_groups = [
                    group for group in ownership_groups
                    if any(
                        str(target.get("target_id") or "") in allowed_target_ids
                        for target in group.targets
                        if isinstance(target, dict)
                    )
                ]
            else:
                ownership_groups = [
                    group for group in ownership_groups
                    if group.operation == strategy_id
                ]
        ownership_groups = rank_scanner_target_groups(
            report=current_report,
            goal=current_goal,
            groups=ownership_groups,
        )[:config.groups_per_round]
        if any(group.operation == "paragraph_preserving_broad_reconstruction" for group in ownership_groups):
            text, trace = _generate_paragraph_ownership_candidate(
                original_text=current_text,
                scan_contract=current_contract,
                content_mode=content_mode,
                family=family,
                target_groups=ownership_groups,
                api_key=api_key,
                model=model,
                base_url=base_url,
            )
            return text, {
                **trace,
                "scanner_controlled": True,
                "planned_strategy_id": strategy_id,
                "ownership_repair_mode": True,
                "executed_strategy": "claim_ownership_repair",
            }

    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(900))

    for round_index in range(1, max(1, int(config.max_rounds)) + 1):
        current_contract = build_scan_contract(current_report, current_text)
        groups = group_rewrite_targets(
            original_text=current_text,
            rewrite_target_profile=current_contract.rewrite_target_profile,
            max_groups=max(_max_target_executor_groups(), config.groups_per_round * 2),
        )
        if strategy_id:
            allowed_target_ids: set[str] = set()
            for problem_group in current_contract.problem_groups or []:
                if not isinstance(problem_group, dict):
                    continue
                allowed = {str(item) for item in problem_group.get("allowed_operations") or []}
                if strategy_id not in allowed:
                    continue
                allowed_target_ids.update(
                    str(target_id)
                    for target_id in problem_group.get("target_ids") or []
                    if str(target_id)
                )
            if allowed_target_ids:
                groups = [
                    group for group in groups
                    if any(
                        str(target.get("target_id") or "") in allowed_target_ids
                        for target in group.targets
                        if isinstance(target, dict)
                    )
                ]
            else:
                groups = [
                    group for group in groups
                    if group.operation == strategy_id
                ]
        groups = rank_scanner_target_groups(
            report=current_report,
            goal=current_goal,
            groups=groups,
        )
        groups = [
            group for group in groups
            if str(group.unit_id or group.group_id) not in accepted_unit_ids
        ][:config.groups_per_round]
        last_groups = groups
        round_log: dict[str, Any] = {
            "round": round_index,
            "start_metrics": current_metrics,
            "start_rank": scanner_controlled_rank(current_metrics),
            "groups": [],
        }
        improved_this_round = False
        if not groups:
            round_log["stop_reason"] = "no_scanner_target_groups"
            round_trace.append(round_log)
            break

        for group in groups:
            prompt = build_scanner_controlled_prompt(
                report=current_report,
                group=group,
                variants_per_group=config.variants_per_group,
                ownership_repair_mode=ownership_repair_mode,
            )
            llm_error: str | None = None
            raw = ""
            try:
                response = gateway.chat(
                    prompt,
                    system="Return only valid JSON with a variants array.",
                    response_format={"type": "json_object"},
                )
                raw = response.content
                provider = response.raw.get("provider")
            except Exception as exc:
                llm_error = str(exc)
                provider = None
                errors.append(llm_error)
            variants, parse_diagnostics = parse_scanner_controlled_variants_with_diagnostics(
                raw,
                limit=config.variants_per_group,
            )
            group_log: dict[str, Any] = {
                "group_id": group.group_id,
                "unit_id": group.unit_id,
                "operation": group.operation,
                "provider": provider,
                "variant_count": len(variants),
                "error": llm_error,
                "parse_diagnostics": parse_diagnostics,
                "variants": [],
            }
            best: dict[str, Any] | None = None
            action_contract = group_action_contract(
                group=group,
                predictability_briefs=predictability_briefs_from_report(current_report),
            )
            for variant_index, variant in enumerate(variants, start=1):
                replacement_text = restore_protected_anchor_placeholders(
                    str(variant.get("replacement_text") or "").strip(),
                    group,
                )
                variant_gate = scanner_controlled_variant_gate(
                    report=current_report,
                    group=group,
                    variant=variant,
                    replacement_text=replacement_text,
                    require_ownership=ownership_repair_mode,
                )
                if not variant_gate.get("passed"):
                    group_log["variants"].append({
                        "variant_index": variant_index,
                        "variant_id": variant.get("variant_id"),
                        "delta": None,
                        "word_count": word_count(replacement_text),
                        "metrics": None,
                        "apply_status": [],
                        "rejected_reason": variant_gate.get("reason"),
                        "variant_gate": variant_gate,
                    })
                    continue
                candidate_quality = scanner_controlled_candidate_quality(
                    action_contract=action_contract,
                    variant_gate=variant_gate,
                    source_text=group.source_text,
                    replacement_text=replacement_text,
                    variant=variant,
                )
                missing_anchors = _missing_protected_anchors(replacement_text, group)
                if missing_anchors:
                    group_log["variants"].append({
                        "variant_index": variant_index,
                        "variant_id": variant.get("variant_id"),
                        "delta": None,
                        "word_count": word_count(replacement_text),
                        "metrics": None,
                        "apply_status": [],
                        "rejected_reason": "protected_anchor_missing",
                        "missing_protected_anchors": missing_anchors,
                        "variant_gate": variant_gate,
                        "candidate_quality": candidate_quality,
                    })
                    continue
                candidate_text, candidate_apply_status = apply_target_replacements(
                    original_text=current_text,
                    target_groups=[group],
                    replacements=[{
                        "group_id": group.group_id,
                        "replacement_text": replacement_text,
                    }],
                )
                candidate_report = _scan_report(candidate_text)
                candidate_goal = evaluate_rewrite_goal(
                    original_text=original_text,
                    candidate_text=candidate_text,
                    original_report=original_report,
                    candidate_report=candidate_report,
                ).to_dict()
                candidate_metrics = scanner_controlled_metrics(
                    report=candidate_report,
                    goal=candidate_goal,
                    footprint_risk=_footprint_risk(_ai_footprint_profile(candidate_report)),
                    ai_score=_badge_ai(candidate_report),
                    topk_score=_topk(candidate_report),
                )
                delta = round(scanner_controlled_rank(current_metrics) - scanner_controlled_rank(candidate_metrics), 3)
                row = {
                    "variant_index": variant_index,
                    "variant_id": variant.get("variant_id"),
                    "delta": delta,
                    "word_count": word_count(replacement_text),
                    "metrics": candidate_metrics,
                    "apply_status": candidate_apply_status,
                    "replacement_text": replacement_text,
                    "text": candidate_text,
                    "report": candidate_report,
                    "goal": candidate_goal,
                    "variant_gate": variant_gate,
                    "candidate_quality": candidate_quality,
                }
                group_log["variants"].append({
                    key: row[key]
                    for key in ("variant_index", "variant_id", "delta", "word_count", "metrics", "apply_status", "variant_gate", "candidate_quality")
                })
                if (
                    best is None
                    or scanner_controlled_rank(candidate_metrics) < scanner_controlled_rank(best["metrics"])
                    or (
                        scanner_controlled_rank(candidate_metrics) == scanner_controlled_rank(best["metrics"])
                        and _number(candidate_quality.get("score")) > _number((best.get("candidate_quality") or {}).get("score"))
                    )
                ):
                    best = row

            if best and _number(best.get("delta")) >= config.min_accept_delta:
                current_text = str(best["text"])
                current_report = best["report"]
                current_goal = best["goal"]
                current_metrics = best["metrics"]
                accepted_row = {
                    "round": round_index,
                    "group_id": group.group_id,
                    "unit_id": group.unit_id,
                    "variant_index": best["variant_index"],
                    "delta": best["delta"],
                    "replacement_text": best["replacement_text"],
                    "candidate_quality": best.get("candidate_quality"),
                    "metrics_after": current_metrics,
                }
                accepted.append(accepted_row)
                accepted_unit_ids.add(str(group.unit_id or group.group_id))
                apply_statuses.extend(best.get("apply_status") or [])
                group_log["accepted_variant"] = best["variant_index"]
                group_log["accepted_delta"] = best["delta"]
                group_log["accepted_candidate_quality"] = best.get("candidate_quality")
                improved_this_round = True
                round_log["groups"].append(group_log)
                break
            elif best:
                group_log["rejected_best_delta"] = best.get("delta")
            else:
                group_log["rejected_best_delta"] = None
            round_log["groups"].append(group_log)

        round_log["end_metrics"] = current_metrics
        round_log["end_rank"] = scanner_controlled_rank(current_metrics)
        round_trace.append(round_log)
        if not improved_this_round:
            break

    accepted_replacements = [
        {
            "group_id": row["group_id"],
            "unit_id": row["unit_id"],
            "replacement_text": row["replacement_text"],
            "round": row["round"],
            "delta": row["delta"],
            "candidate_quality": row.get("candidate_quality"),
        }
        for row in accepted
    ]
    trace = target_execution_trace(
        attempted=True,
        target_groups=last_groups,
        replacements=accepted_replacements,
        apply_status=apply_statuses,
        batches=[],
        error="; ".join(errors) if errors else None,
    )
    trace = {
        **trace,
        "scanner_controlled": True,
        "planned_strategy_id": strategy_id,
        "ownership_repair_mode": bool(ownership_repair_mode),
        "scanner_controlled_config": config.to_dict(),
        "scanner_controlled_rounds": round_trace,
        "scanner_controlled_accepted": accepted_replacements,
        "scanner_controlled_initial_metrics": scanner_controlled_metrics(
            report=original_report,
            goal=original_goal,
            footprint_risk=_footprint_risk(_ai_footprint_profile(original_report)),
            ai_score=_badge_ai(original_report),
            topk_score=_topk(original_report),
        ),
        "scanner_controlled_final_metrics": current_metrics,
        "scanner_controlled_rank_delta": round(
            scanner_controlled_rank(scanner_controlled_metrics(
                report=original_report,
                goal=original_goal,
                footprint_risk=_footprint_risk(_ai_footprint_profile(original_report)),
                ai_score=_badge_ai(original_report),
                topk_score=_topk(original_report),
            ))
            - scanner_controlled_rank(current_metrics),
            3,
        ),
    }
    if not accepted:
        trace = {**trace, "error": trace.get("error") or "no_scanner_controlled_positive_variant"}
        return "", trace
    return clean_v3_candidate_output(current_text), trace


def _generate_assisted_footprint_candidate(
    *,
    original_text: str,
    original_report: dict[str, Any],
    scan_contract: ScanContract,
    content_mode: str,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> tuple[str, dict[str, Any]]:
    groups = group_assisted_footprint_windows(
        original_text=original_text,
        authorship_window_profile=_authorship_window_profile(original_report),
        rewrite_target_profile=scan_contract.rewrite_target_profile,
        max_groups=_max_assisted_footprint_groups(),
    )
    if not groups:
        return "", target_execution_trace(
            attempted=True,
            target_groups=[],
            error="no_assisted_footprint_groups",
        )
    replacements: list[dict[str, str]] = []
    batch_trace: list[dict[str, Any]] = []
    batch_errors: list[str] = []
    for batch_index, batch in enumerate(batch_target_groups(groups, batch_size=_assisted_footprint_batch_size()), start=1):
        prompt = build_assisted_footprint_prompt(
            target_groups=batch,
            content_mode=content_mode,
            predictability_briefs=scan_contract.predictability_briefs,
        )
        target_words = sum(int(group.word_count_guide.get("preferred_words") or word_count(group.source_text)) for group in batch)
        gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(max(260, target_words + 520)))
        batch_error = None
        try:
            raw = gateway.chat(
                prompt,
                system="Return only valid JSON with a replacements array.",
                response_format={"type": "json_object"},
            ).content
            batch_replacements = parse_target_replacements(raw)
        except Exception as exc:
            batch_error = str(exc)
            batch_errors.append(batch_error)
            batch_replacements = []
        replacements.extend(batch_replacements)
        batch_trace.append({
            "batch_index": batch_index,
            "group_ids": [group.group_id for group in batch],
            "requested_groups": len(batch),
            "replacement_count": len(batch_replacements),
            "error": batch_error,
        })
    if not replacements:
        return "", target_execution_trace(
            attempted=True,
            target_groups=groups,
            replacements=[],
            batches=batch_trace,
            error=batch_errors[0] if batch_errors else "generation_failed_empty_output",
        )
    text, apply_status = apply_assisted_footprint_replacements(
        original_text=original_text,
        target_groups=groups,
        replacements=replacements,
    )
    trace = target_execution_trace(
        attempted=True,
        target_groups=groups,
        replacements=replacements,
        apply_status=apply_status,
        batches=batch_trace,
        error="; ".join(batch_errors) if batch_errors else None,
    )
    if text.strip() == original_text.strip():
        trace = {**trace, "error": "no_assisted_footprint_replacement_applied"}
    return clean_v3_candidate_output(text), trace


def _generate_authorship_window_repair_candidate(
    *,
    candidate_text: str,
    candidate_trace: dict[str, Any],
    family: str,
    contract: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    profile = candidate_trace.get("authorship_window_profile") if isinstance(candidate_trace.get("authorship_window_profile"), dict) else {}
    target_windows = select_authorship_window_targets(
        profile,
        max_targets=_max_authorship_window_repair_targets(),
    )
    if not target_windows:
        return str(candidate_text or "")
    prompt = build_authorship_window_repair_prompt(
        candidate_text=candidate_text,
        target_windows=target_windows,
        strategy_family=family,
        contract=contract,
    )
    target_words = sum(int(window.get("word_count") or 0) for window in target_windows)
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(max(120, target_words + 120)))
    raw = gateway.chat(prompt, system="Return only valid JSON with a replacements array.").content
    replacements = extract_authorship_window_replacements(raw)
    if not replacements:
        return str(candidate_text or "")
    return clean_v3_candidate_output(apply_authorship_window_replacements(
        candidate_text=candidate_text,
        target_windows=target_windows,
        replacements=replacements,
    ))


def _split_success_pair(candidate_evaluations: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    detector_best: dict[str, Any] | None = None
    ownership_best: dict[str, Any] | None = None

    def detector_score(item: dict[str, Any]) -> tuple[float, float, float]:
        trace = item.get("trace") if isinstance(item.get("trace"), dict) else {}
        proxy = trace.get("external_proxy") if isinstance(trace.get("external_proxy"), dict) else {}
        metrics = proxy.get("metrics") if isinstance(proxy.get("metrics"), dict) else {}
        ai_delta = metrics.get("ai_delta") if isinstance(metrics.get("ai_delta"), (int, float)) else 0.0
        topk_delta = metrics.get("topk_delta") if isinstance(metrics.get("topk_delta"), (int, float)) else 0.0
        candidate_ai = trace.get("candidate_ai") if isinstance(trace.get("candidate_ai"), (int, float)) else 100.0
        return (float(ai_delta), float(topk_delta), -float(candidate_ai))

    def ownership_score(item: dict[str, Any]) -> tuple[float, float]:
        gate = ((item.get("trace") or {}).get("ownership_gate") if isinstance(item.get("trace"), dict) else {}) or {}
        return (
            float(gate.get("ownership_score") or 0.0) if isinstance(gate.get("ownership_score"), (int, float)) else 0.0,
            float(gate.get("ownership_change_count") or 0.0) if isinstance(gate.get("ownership_change_count"), (int, float)) else 0.0,
        )

    for item in candidate_evaluations:
        text = str(item.get("text") or "").strip()
        trace = item.get("trace") if isinstance(item.get("trace"), dict) else {}
        if not text or trace.get("validity_status") != "valid":
            continue
        gate = trace.get("ownership_gate") if isinstance(trace.get("ownership_gate"), dict) else {}
        if bool(trace.get("detector_movement")) and bool(gate.get("active")) and not bool(gate.get("passed")):
            if detector_best is None or detector_score(item) > detector_score(detector_best):
                detector_best = item
        if bool(gate.get("active")) and bool(gate.get("passed")):
            if ownership_best is None or ownership_score(item) > ownership_score(ownership_best):
                ownership_best = item
    return detector_best, ownership_best


def _generate_detector_ownership_fusion_candidate(
    *,
    source_text: str,
    candidate_evaluations: list[dict[str, Any]],
    family: str,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> tuple[str, dict[str, Any]]:
    detector_item, ownership_item = _split_success_pair(candidate_evaluations)
    if not detector_item or not ownership_item:
        return "", {
            "target_execution_attempted": True,
            "executor_engine": "detector_ownership_fusion",
            "executed_strategy": "fuse_detector_and_ownership",
            "error": "split_success_pair_not_found",
        }
    prompt = build_detector_ownership_fusion_prompt(
        source_text=source_text,
        detector_candidate=str(detector_item.get("text") or ""),
        ownership_candidate=str(ownership_item.get("text") or ""),
        detector_trace=detector_item.get("trace") if isinstance(detector_item.get("trace"), dict) else {},
        ownership_trace=ownership_item.get("trace") if isinstance(ownership_item.get("trace"), dict) else {},
        family=family,
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(max(word_count(str(detector_item.get("text") or "")), word_count(source_text)) + 420))
    response = gateway.chat(
        prompt,
        system="Return only valid JSON with rewritten_document.",
        response_format={"type": "json_object"},
    )
    raw_content = _reject_length_limited_raw_response(response, stage="detector_ownership_fusion")
    fused, extraction_diagnostics = extract_fused_document_with_diagnostics(raw_content)
    response_diagnostics = {
        "finish_reason": getattr(response, "finish_reason", None),
        "native_finish_reason": getattr(response, "native_finish_reason", None),
        "content_chars": len(str(getattr(response, "content", "") or "")),
        "content_preview": str(getattr(response, "content", "") or "")[:1200],
        "raw_content_chars": len(str(getattr(response, "raw_content", "") or "")),
        "raw_content_preview": str(getattr(response, "raw_content", "") or "")[:1200],
        "extraction": extraction_diagnostics,
    }
    return clean_v3_candidate_output(fused), {
        "target_execution_attempted": True,
        "executor_engine": "detector_ownership_fusion",
        "executed_strategy": "fuse_detector_and_ownership",
        "prompt_stage": "detector_ownership_fusion",
        "detector_source_mode": (detector_item.get("trace") or {}).get("generation_mode"),
        "ownership_source_mode": (ownership_item.get("trace") or {}).get("generation_mode"),
        "detector_source_ai": (detector_item.get("trace") or {}).get("candidate_ai"),
        "ownership_source_ai": (ownership_item.get("trace") or {}).get("candidate_ai"),
        "detector_source_ownership_gate": (detector_item.get("trace") or {}).get("ownership_gate"),
        "ownership_source_ownership_gate": (ownership_item.get("trace") or {}).get("ownership_gate"),
        "fusion_response_diagnostics": response_diagnostics,
        "error": None if fused else "empty_fusion_output",
    }


def _generate_structure_repair_candidate(
    *,
    original_text: str,
    candidate_text: str,
    validation: dict[str, Any],
    expected_unit_count: int | None,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
) -> str:
    prompt = build_structure_repair_prompt(
        source_text=original_text,
        candidate_text=candidate_text,
        validation=validation,
        expected_unit_count=expected_unit_count,
    )
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(word_count(candidate_text)))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the repaired text as plain text.").content)


def _unit_chunks(source_units: list[dict[str, Any]] | str, *, force_unit_chunks: bool = False) -> list[list[dict[str, Any]]]:
    if isinstance(source_units, str):
        units_payload = [unit.to_dict() for unit in document_units(source_units)]
    else:
        units_payload = source_units
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_words = 0
    limit = _chunk_word_limit()
    for payload in units_payload:
        unit_words = int(payload.get("word_count") or word_count(str(payload.get("text") or "")))
        if force_unit_chunks:
            chunks.append([payload])
            continue
        if current and current_words + unit_words > limit:
            chunks.append(current)
            current = []
            current_words = 0
        current.append(payload)
        current_words += unit_words
    if current:
        chunks.append(current)
    return chunks


def _normalize_chunk_unit_boundaries(text: str, *, expected_units: int) -> str:
    units = document_units(text)
    if expected_units == 1 and len(units) > 1:
        return "\n".join(unit.text.strip() for unit in units if unit.text.strip()).strip()
    return str(text or "").strip()


def _generate_chunked_candidate(
    *,
    original_text: str,
    source_units: list[dict[str, Any]],
    family: str,
    contract: Any,
    compression_policy: Any,
    api_key: str | None,
    model: str | None,
    base_url: str | None,
    rewrite_target_profile: dict[str, Any] | None = None,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    central_judgment_plan: dict[str, Any] | None = None,
    force_unit_chunks: bool = False,
) -> str:
    examples = examples_for_family(family)
    global_plan = {
        "family": family,
        "source_words": word_count(original_text),
        "source_units": len(source_units),
        "compression_policy": compression_policy.to_dict(),
        "instruction": "Keep chunk outputs compatible when joined; do not add global conclusions unless present in the chunk.",
    }
    rewritten_chunks: list[str] = []
    for chunk in _unit_chunks(source_units, force_unit_chunks=force_unit_chunks):
        chunk_words = sum(int(unit.get("word_count") or 0) for unit in chunk)
        chunk_policy = compression_policy_for_family(family, chunk_words)
        if family == "cited_practice_voice":
            prompt = build_cited_practice_voice_chunk_prompt(
                source_units=chunk,
                contract=contract,
                global_plan=global_plan,
                compression_policy=chunk_policy,
                style_examples=examples,
                rewrite_target_profile=rewrite_target_profile,
                predictability_briefs=predictability_briefs,
                central_judgment_plan=central_judgment_plan,
            )
        elif family == "clean_texture_boundary":
            prompt = build_clean_texture_boundary_chunk_prompt(
                source_units=chunk,
                global_plan=global_plan,
                style_examples=examples,
                rewrite_target_profile=rewrite_target_profile,
                predictability_briefs=predictability_briefs,
                central_judgment_plan=central_judgment_plan,
            )
        else:
            prompt = build_document_rhythm_chunk_prompt(
                source_units=chunk,
                global_plan=global_plan,
                compression_policy=chunk_policy,
                style_examples=examples,
                rewrite_target_profile=rewrite_target_profile,
                predictability_briefs=predictability_briefs,
                central_judgment_plan=central_judgment_plan,
            )
        token_words = chunk_words if family == "clean_texture_boundary" else chunk_policy.max_words
        gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(token_words))
        kwargs = _clean_texture_chat_kwargs() if family == "clean_texture_boundary" else {}
        rewritten_chunk = clean_v3_candidate_output(gateway.chat(
            prompt,
            system="Return only rewritten plain text for this chunk.",
            **kwargs,
        ).content)
        rewritten_chunk = _normalize_chunk_unit_boundaries(rewritten_chunk, expected_units=len(chunk))
        rewritten_chunks.append(_restore_exact_quote_anchors(rewritten_chunk, contract))
    return _restore_exact_quote_anchors(compose_units(rewritten_chunks), contract)


def _candidate_from_replay(record: dict[str, Any], original_report: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    text = str(record.get("text") or record.get("candidate_text") or "").strip()
    report = record.get("report") if isinstance(record.get("report"), dict) else None
    if report is None and isinstance(record.get("ai"), (int, float)):
        report = {
            "ai_risk_badge": {
                "ai_likelihood_score": record.get("ai"),
                "writing_quality_score": record.get("wq") or record.get("writing_quality"),
                "ai_components": {"topk_pattern_raw": record.get("topk")},
            },
            "findings": {"critical": [], "high": [], "medium": [], "low": []},
        }
    return text, report


def run_rewrite_pipeline_v3(
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

    def progress(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    progress(62, "Starting external-calibrated rewrite V3")
    original_text = _extract_original_text(detect_json)
    original_report = detect_json
    scan_contract = build_scan_contract(original_report, original_text)
    v3_route = route_from_scan_contract(scan_contract)
    strategy_plan = build_strategy_plan(v3_route, scan_contract)
    content_mode, content_router_trace = _v3_content_mode(scan_contract, v3_route)
    contract = build_rewrite_contract(original_text, content_mode=content_mode)
    central_judgment_plan = build_contextual_judgment_plan(
        source_text=original_text,
        scan_report=original_report,
    ).to_dict()
    exact_anchor_count = sum(1 for anchor in contract.anchors if anchor.severity.value == "hard_exact")
    family = _family_for_route(content_mode, exact_anchor_count)
    compression_policy = compression_policy_for_family(family, word_count(original_text))
    source_generation_units = _logical_source_units(original_text, original_report)
    force_unit_chunks = _should_force_unit_chunks(
        scan_contract=scan_contract,
        v3_route=v3_route,
        exact_anchor_count=exact_anchor_count,
    )
    generation_mode = "replay"
    candidate_text = ""
    candidate_report: dict[str, Any] | None = None
    generation_error = None
    target_execution = target_execution_trace(
        attempted=False,
        target_groups=group_rewrite_targets(
            original_text=original_text,
            rewrite_target_profile=scan_contract.rewrite_target_profile,
            max_groups=_max_target_executor_groups(),
        ) if _target_executor_available(scan_contract) else [],
    )
    first_strategy_step = _first_strategy_step_id(strategy_plan)
    problem_inventory_driven = bool(scan_contract.problem_groups)

    if replay_candidate_records:
        candidate_text, candidate_report = _candidate_from_replay(replay_candidate_records[0], original_report)
    elif full_rewrite_allowed:
        try:
            if first_strategy_step == "unit_preserving_prune_bridge" and _prune_bridge_available(scan_contract):
                generation_mode = "unit_preserving_prune_bridge"
                candidate_text, target_execution = _generate_prune_bridge_candidate(
                    original_text=original_text,
                    scan_contract=scan_contract,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
                target_execution = _annotate_target_execution(
                    target_execution,
                    executed_strategy="unit_preserving_prune_bridge",
                    executor_engine="unit_preserving_prune_bridge",
                )
            elif (
                problem_inventory_driven
                and not _uses_problem_scanner_controlled_strategy(first_strategy_step)
                and _scanner_controlled_should_run_first(scan_contract)
            ):
                generation_mode = "scanner_controlled_executor"
                candidate_text, target_execution = _generate_scanner_controlled_candidate(
                    original_text=original_text,
                    original_report=original_report,
                    scan_contract=scan_contract,
                    content_mode=content_mode,
                    family=family,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
                target_execution = _annotate_target_execution(
                    target_execution,
                    executed_strategy="scanner_controlled_span_repair",
                    executor_engine="scanner_controlled_executor",
                )
            elif (
                problem_inventory_driven
                and _uses_problem_scanner_controlled_strategy(first_strategy_step)
                and _scanner_controlled_executor_available(scan_contract)
            ):
                generation_mode = first_strategy_step
                candidate_text, target_execution = _generate_scanner_controlled_candidate(
                    original_text=original_text,
                    original_report=original_report,
                    scan_contract=scan_contract,
                    content_mode=content_mode,
                    family=family,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
                target_execution = _annotate_target_execution(
                    target_execution,
                    executed_strategy=first_strategy_step,
                    executor_engine="scanner_controlled_executor",
                )
            elif (
                problem_inventory_driven
                and _uses_problem_target_executor_strategy(first_strategy_step)
                and _target_executor_available(scan_contract)
            ):
                generation_mode = first_strategy_step
                candidate_text, target_execution = _generate_target_executor_candidate(
                    original_text=original_text,
                    scan_contract=scan_contract,
                    content_mode=content_mode,
                    family=family,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
                target_execution = _annotate_target_execution(
                    target_execution,
                    executed_strategy=first_strategy_step,
                    executor_engine="target_executor",
                )
            elif problem_inventory_driven and first_strategy_step == "chunk_reconstruction":
                generation_mode = "chunk_reconstruction"
                candidate_text = _generate_chunked_candidate(
                    original_text=original_text,
                    source_units=source_generation_units,
                    family=family,
                    contract=contract,
                    compression_policy=compression_policy,
                    rewrite_target_profile=scan_contract.rewrite_target_profile,
                    predictability_briefs=scan_contract.predictability_briefs,
                    central_judgment_plan=central_judgment_plan,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    force_unit_chunks=force_unit_chunks,
                )
            elif _scanner_controlled_executor_available(scan_contract):
                generation_mode = "scanner_controlled_executor"
                candidate_text, target_execution = _generate_scanner_controlled_candidate(
                    original_text=original_text,
                    original_report=original_report,
                    scan_contract=scan_contract,
                    content_mode=content_mode,
                    family=family,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
                target_execution = _annotate_target_execution(
                    target_execution,
                    executed_strategy="scanner_controlled_executor",
                    executor_engine="scanner_controlled_executor",
                )
            elif _strategy_plan_prefers_target_executor(strategy_plan, scan_contract):
                generation_mode = "target_executor"
                candidate_text, target_execution = _generate_target_executor_candidate(
                    original_text=original_text,
                    scan_contract=scan_contract,
                    content_mode=content_mode,
                    family=family,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
            elif _should_use_chunked_generation(
                source_words=word_count(original_text),
                scan_contract=scan_contract,
                v3_route=v3_route,
                exact_anchor_count=exact_anchor_count,
            ):
                generation_mode = "chunked"
                candidate_text = _generate_chunked_candidate(
                    original_text=original_text,
                    source_units=source_generation_units,
                    family=family,
                    contract=contract,
                    compression_policy=compression_policy,
                    rewrite_target_profile=scan_contract.rewrite_target_profile,
                    predictability_briefs=scan_contract.predictability_briefs,
                    central_judgment_plan=central_judgment_plan,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    force_unit_chunks=force_unit_chunks,
                )
            else:
                generation_mode = "single_shot"
                candidate_text = _generate_single_candidate(
                    original_text=original_text,
                    original_report=original_report,
                    family=family,
                    contract=contract,
                    compression_policy=compression_policy,
                    rewrite_target_profile=scan_contract.rewrite_target_profile,
                    predictability_briefs=scan_contract.predictability_briefs,
                    central_judgment_plan=central_judgment_plan,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                )
        except Exception as exc:
            generation_error = str(exc)
            if generation_mode == "target_executor":
                target_execution = {
                    **target_execution,
                    "target_execution_attempted": True,
                    "error": generation_error,
                }
            if _uses_problem_target_executor_strategy(generation_mode):
                target_execution = {
                    **target_execution,
                    "target_execution_attempted": True,
                    "executed_strategy": generation_mode,
                    "executor_engine": "target_executor",
                    "error": generation_error,
                }
            if generation_mode == "unit_preserving_prune_bridge":
                target_execution = {
                    **target_execution,
                    "target_execution_attempted": True,
                    "problem_strategy": "unit_preserving_prune_bridge",
                    "error": generation_error,
                }
            if generation_mode == "scanner_controlled_executor" or _uses_problem_scanner_controlled_strategy(generation_mode):
                target_execution = {
                    **target_execution,
                    "target_execution_attempted": True,
                    "scanner_controlled": True,
                    "executed_strategy": generation_mode,
                    "executor_engine": "scanner_controlled_executor",
                    "error": generation_error,
                }

    reference_ai = _badge_ai(original_report)
    target_ai_score = float(reference_ai) - float(required_ai_drop) if isinstance(reference_ai, (int, float)) else None
    expected_unit_count = len(source_generation_units)
    original_footprint = _ai_footprint_profile(original_report) or scan_contract.ai_footprint_profile
    original_target_profile = _rewrite_target_profile(original_report) or scan_contract.rewrite_target_profile
    original_problem_inventory = _problem_inventory(original_report) or scan_contract.problem_inventory

    def assess_candidate(
        *,
        text: str,
        report: dict[str, Any] | None,
        mode: str,
        cost: int,
        error: str | None = None,
        target_execution_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        text = _restore_exact_quote_anchors(text, contract)
        validation_result = validate_v3_candidate(
            original_text=original_text,
            candidate_text=text,
            contract=contract,
            require_unit_count=True,
            expected_unit_count=expected_unit_count,
        )
        compression_result = compression_status(original_text, text, compression_policy)
        compression_ok = _compression_accepted(compression_result)
        integrity_result = _text_integrity(original_text, text)
        generated_empty = not bool(str(text or "").strip())
        integrity_passed = bool(integrity_result["passed"])
        should_scan_candidate = bool(not generated_empty and integrity_passed)
        scanned_report = report
        scan_input_hash = _text_hash(text)
        if should_scan_candidate and scanned_report is None:
            progress(78, f"Scanning V3 {mode} candidate")
            scanned_report = _scan_report(text)
        elif scanned_report is None and not generated_empty and integrity_passed:
            scanned_report = original_report
        elif scanned_report is None:
            scanned_report = {}
        report_input_text = str((scanned_report or {}).get("input_text") or "")
        report_input_hash = _text_hash(report_input_text)
        scan_freshness = {
            "candidate_text_hash": scan_input_hash,
            "report_input_text_hash": report_input_hash,
            "input_text_present": bool(report_input_text),
            "input_text_matches_candidate": bool(report_input_text == str(text or "")),
            "scan_reused_supplied_report": bool(report is not None),
            "empty_candidate_no_report_fallback": bool(generated_empty and report is None),
            "integrity_failed_no_scan": bool(not integrity_passed and report is None),
            "predictability_sentence_cache_enabled": os.environ.get("DRAFTPROOF_PREDICTABILITY_SENTENCE_CACHE", "1").lower() not in {"0", "false", "no", "off"},
            "predictability_cache": _predictability_cache_info(scanned_report),
        }
        candidate_footprint = _ai_footprint_profile(scanned_report)
        footprint_delta = _footprint_delta(original_footprint, candidate_footprint)
        candidate_target_profile = _rewrite_target_profile(scanned_report)
        target_movement = _target_profile_movement(original_target_profile, candidate_target_profile)
        candidate_problem_inventory = _problem_inventory(scanned_report)
        unresolved_problem_groups = (
            candidate_problem_inventory.get("problem_groups")
            if isinstance(candidate_problem_inventory.get("problem_groups"), list)
            else []
        )
        target_trace = target_execution_info if isinstance(target_execution_info, dict) else {}
        topk_effect_failure_rows = _topk_effect_failure_rows(target_trace)
        topk_effect_failures = _topk_effect_failures(target_trace)
        unapplied_target_group_ids = _unapplied_target_group_ids(target_trace)
        target_gate_passed = bool(target_movement["moved"] and not unapplied_target_group_ids)
        central_profile_score = score_candidate_against_central_profile(
            scanned_report,
            baseline_report=original_report,
        )
        goal_result = evaluate_rewrite_goal(
            original_text=original_text,
            candidate_text=text or original_text,
            original_report=original_report,
            candidate_report=scanned_report,
        )
        semantic_result = check_semantic_drift(original_text, text or original_text, threshold=0.15)
        semantic_safe = _semantic_safe_for_v3(semantic_result)
        decision_result = decide_candidate(
            goal=goal_result,
            original_report=original_report,
            candidate_report=scanned_report,
            reference_ai=reference_ai,
            required_ai_drop=required_ai_drop,
            target_ai_score=target_ai_score,
            semantic_safe=semantic_safe,
            quality_safe=bool(
                validation_result.passed
                and compression_ok
                and integrity_result["passed"]
                and footprint_delta["moved"]
                and target_gate_passed
            ),
            cost=cost,
        )
        validity = _validity_status(
            validation_passed=bool(validation_result.passed),
            compression_ok=bool(compression_ok),
            semantic_safe=semantic_safe,
            integrity_passed=bool(integrity_result["passed"]),
        )
        generation_failure_outcome = _generation_failure_outcome(text=text, error=error)
        outcome = generation_failure_outcome or _candidate_outcome(
            validity_status=validity,
            detector_moved=bool(footprint_delta["moved"]),
            integrity_passed=bool(integrity_result["passed"]),
        )
        proxy_result = evaluate_external_proxy(
            family=family,
            reference_ai=reference_ai,
            candidate_ai=_badge_ai(scanned_report),
            reference_wq=_badge_wq(original_report),
            candidate_wq=_badge_wq(scanned_report),
            reference_topk=_topk(original_report),
            candidate_topk=_topk(scanned_report),
            candidate_authorship_profile=_authorship_window_profile(scanned_report),
            compression=compression_result,
            validation_passed=bool(validation_result.passed),
            compression_accepted=bool(compression_ok),
            semantic_safe=semantic_safe,
        )
        ownership_gate = _ownership_gate(
            proxy_result=proxy_result,
            target_trace=target_trace,
            goal_result=goal_result,
        )
        can_select = bool(
            text
            and validity == "valid"
            and footprint_delta["moved"]
            and target_gate_passed
            and integrity_result["passed"]
            and ownership_gate["passed"]
        )
        target_attempted = bool(target_trace.get("target_execution_attempted"))
        return {
            "text": text,
            "report": scanned_report,
            "should_scan": should_scan_candidate,
            "strict_selected": bool(can_select and decision_result.lane == CandidateLane.GOAL_MET),
            "external_selected": bool(can_select and proxy_result.accepted),
            "trace": {
                "strategy_family": family,
                "generation_mode": mode,
                "candidate_ai": _badge_ai(scanned_report),
                "candidate_wq": _badge_wq(scanned_report),
                "candidate_topk": _topk(scanned_report),
                "scan_freshness": scan_freshness,
                "authorship_window_profile": _authorship_window_profile(scanned_report),
                "footprint_before": original_footprint,
                "footprint_after": candidate_footprint,
                "footprint_delta": footprint_delta,
                "target_profile_before": original_target_profile,
                "target_profile_after": candidate_target_profile,
                "target_movement": target_movement,
                "target_gate_passed": target_gate_passed,
                "unapplied_target_group_ids": unapplied_target_group_ids,
                "problem_inventory_before": original_problem_inventory,
                "problem_inventory_after": candidate_problem_inventory,
                "unresolved_problem_groups": unresolved_problem_groups,
                "target_execution_available": _target_executor_available(scan_contract),
                "scanner_controlled_executor_available": _scanner_controlled_executor_available(scan_contract),
                "prune_bridge_available": _prune_bridge_available(scan_contract),
                "assisted_footprint_executor_available": _assisted_footprint_executor_available(scan_contract),
                "target_execution_attempted": target_attempted,
                "target_execution_trace": target_trace,
                "prompt_template_id": target_trace.get("prompt_template_id") if target_trace else None,
                "prompt_stage": target_trace.get("prompt_stage") if target_trace else None,
                "scanner_context_used": target_trace.get("scanner_context_used") if target_trace else [],
                "planner_output": target_trace.get("planner_output") if target_trace else None,
                "stage_apply_status": target_trace.get("stage_apply_status") if target_trace else [],
                "stage_rescan_delta": target_trace.get("stage_rescan_delta") if target_trace else None,
                "topk_repair_attempted": bool(target_trace.get("topk_repair_attempted")) if target_trace else False,
                "topk_effect_failures": topk_effect_failures,
                "topk_effect_failure_rows": topk_effect_failure_rows,
                "strategy_stop_reason": target_trace.get("strategy_stop_reason") if target_trace else None,
                "target_groups": target_trace.get("target_groups") if target_trace else [],
                "target_replacements": target_trace.get("target_replacements") if target_trace else [],
                "target_apply_status": target_trace.get("target_apply_status") if target_trace else [],
                "target_rescan_delta": {
                    "footprint_delta": footprint_delta,
                    "target_movement": target_movement,
                },
                "unresolved_targets": target_trace.get("unresolved_targets") if target_trace else [],
                "scanner_intelligence_execution": (
                    "executed_target_profile"
                    if target_attempted
                    else "prompt_context_only"
                    if scan_contract.rewrite_targets
                    else "not_available"
                ),
                "central_judgment_plan": central_judgment_plan,
                "central_profile_score": central_profile_score,
                "detector_movement": bool(footprint_delta["moved"]),
                "candidate_outcome": outcome,
                "validity_status": validity,
                "text_integrity": integrity_result,
                "why_not_success": [
                    reason for reason, failed in (
                        ("detector_footprint_not_improved", not footprint_delta["moved"]),
                        ("rewrite_targets_not_improved", not target_movement["moved"]),
                        ("rewrite_targets_not_applied", bool(unapplied_target_group_ids)),
                        ("central_profile_not_improved", _number(central_profile_score.get("weighted_delta")) <= 0.0),
                        ("validation_failed", not validation_result.passed),
                        ("compression_rejected", not compression_ok),
                        ("semantic_drift", not semantic_safe),
                        ("text_integrity_failed", not integrity_result["passed"]),
                        ("ownership_gate_failed", not ownership_gate["passed"]),
                    )
                    if failed
                ],
                "validation": validation_result.to_dict(),
                "compression": compression_result,
                "compression_accepted": compression_ok,
                "goal": goal_result.to_dict(),
                "decision": decision_result.to_dict(),
                "semantic_safe": semantic_safe,
                "semantic_accepted_raw": bool(getattr(semantic_result, "accepted", False)),
                "semantic_reasons": list(getattr(semantic_result, "reasons", []) or []),
                "semantic_similarity": getattr(semantic_result, "similarity", None),
                "external_proxy": proxy_result.to_dict(),
                "ownership_gate": ownership_gate,
                "generation_error": error,
            },
        }

    candidate_evaluations: list[dict[str, Any]] = []
    candidate_evaluations.append(assess_candidate(
        text=candidate_text,
        report=candidate_report,
        mode=generation_mode,
        cost=1,
        error=generation_error,
        target_execution_info=target_execution,
    ))
    loop_trace = []
    tried_actions: set[CandidateAction] = set()
    tried_problem_strategy_ids: set[str] = set()
    if generation_mode:
        tried_problem_strategy_ids.add(str(generation_mode))
    if generation_mode == "target_executor":
        tried_actions.add(CandidateAction.TARGET_EXECUTOR)
    if _uses_problem_target_executor_strategy(generation_mode):
        tried_actions.add(CandidateAction.TARGET_EXECUTOR)
    if generation_mode == "scanner_controlled_executor" or _uses_problem_scanner_controlled_strategy(generation_mode):
        tried_actions.add(CandidateAction.TARGET_EXECUTOR)
        tried_actions.add(CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR)
    primary_generation_failed = bool(
        not candidate_text
        and (
            generation_error
            or (isinstance(target_execution, dict) and str(target_execution.get("error") or "").startswith("generation_failed"))
        )
    )
    if replay_candidate_records and len(replay_candidate_records) > 1:
        for index, record in enumerate(replay_candidate_records[1:], start=2):
            replay_text, replay_report = _candidate_from_replay(record, original_report)
            candidate_evaluations.append(assess_candidate(
                text=replay_text,
                report=replay_report,
                mode=f"replay_recovery_{index}",
                cost=index,
            ))
    elif full_rewrite_allowed and not primary_generation_failed:
        while (time.time() - started) < max_runtime_seconds and len(candidate_evaluations) < 5:
            latest_trace = candidate_evaluations[-1].get("trace") if isinstance(candidate_evaluations[-1].get("trace"), dict) else {}
            latest_issues = issues_from_trace(latest_trace)
            ownership_repair_due = (
                CandidateIssue.OWNERSHIP_MISSING in latest_issues
                and CandidateAction.CLAIM_OWNERSHIP_REPAIR not in tried_actions
                and bool(latest_trace.get("scanner_controlled_executor_available") or latest_trace.get("target_execution_available"))
            )
            if ownership_repair_due:
                planned_problem_strategy = ""
                loop_decision = LoopDecision(
                    action=CandidateAction.CLAIM_OWNERSHIP_REPAIR,
                    source_index=len(candidate_evaluations) - 1,
                    issues=latest_issues,
                    reason="claim_ownership_repair_before_problem_strategy_exhaustion",
                )
            else:
                planned_problem_strategy = _next_planned_problem_strategy(
                    strategy_plan=strategy_plan,
                    tried_strategy_ids=tried_problem_strategy_ids,
                    latest_trace=latest_trace,
                )
            if planned_problem_strategy:
                loop_decision = LoopDecision(
                    action=CandidateAction.TARGET_EXECUTOR,
                    source_index=len(candidate_evaluations) - 1,
                    issues=issues_from_trace(latest_trace),
                    reason=f"execute_planned_problem_strategy:{planned_problem_strategy}",
                )
            elif not ownership_repair_due:
                exhausted_scanner_problem = (
                    problem_inventory_driven
                    and any(
                        issue in latest_issues
                        for issue in (
                            CandidateIssue.NO_DETECTOR_MOVEMENT,
                            CandidateIssue.NO_TARGET_MOVEMENT,
                            CandidateIssue.GENERATION_FAILED,
                        )
                    )
                )
                fallback_loop_decision = decide_next_action(
                    candidate_evaluations,
                    has_positive_boundaries=bool(examples_for_family(family).get("positive") or []),
                    tried_actions=tried_actions,
                )
                if fallback_loop_decision.action == CandidateAction.FUSE_DETECTOR_AND_OWNERSHIP:
                    loop_decision = fallback_loop_decision
                elif exhausted_scanner_problem:
                    loop_decision = LoopDecision(
                        action=CandidateAction.RETURN_BEST_FOR_REVIEW,
                        source_index=len(candidate_evaluations) - 1,
                        issues=latest_issues,
                        reason="stop_after_problem_strategy_exhausted",
                    )
                else:
                    loop_decision = fallback_loop_decision
                if (
                    loop_decision.action == CandidateAction.REPAIR_TARGETED
                    and not _unbounded_recovery_enabled()
                ):
                    loop_decision = LoopDecision(
                        action=CandidateAction.RETURN_BEST_FOR_REVIEW,
                        source_index=loop_decision.source_index,
                        issues=loop_decision.issues,
                        reason="stop_before_unbounded_recovery_revision",
                    )
            loop_trace.append(loop_decision.to_dict())
            if loop_decision.action in {
                CandidateAction.ACCEPT_STRICT,
                CandidateAction.ACCEPT_EXTERNAL,
                CandidateAction.RETURN_BEST_FOR_REVIEW,
            }:
                break
            tried_actions.add(loop_decision.action)
            if planned_problem_strategy:
                tried_problem_strategy_ids.add(planned_problem_strategy)
            source_item = candidate_evaluations[loop_decision.source_index]
            try:
                loop_target_execution: dict[str, Any] | None = None
                if loop_decision.action == CandidateAction.REPAIR_STRUCTURE:
                    progress(84, "Repairing V3 candidate structure")
                    new_text = _generate_structure_repair_candidate(
                        original_text=original_text,
                        candidate_text=str(source_item.get("text") or ""),
                        validation=source_item["trace"]["validation"],
                        expected_unit_count=expected_unit_count,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "structure_repair"
                elif loop_decision.action == CandidateAction.REPAIR_CONTRACT:
                    progress(81, "Repairing V3 candidate contract")
                    new_text = _generate_contract_repair_candidate(
                        original_text=original_text,
                        failed_candidate=str(source_item.get("text") or candidate_text),
                        family=family,
                        candidate_trace=source_item["trace"],
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "contract_repair"
                elif loop_decision.action == CandidateAction.ADAPT_BOUNDARY:
                    progress(82, "Running V3 boundary adapter")
                    new_text = _generate_boundary_candidate(
                        original_text=original_text,
                        failed_candidates=[str(item.get("text") or "") for item in candidate_evaluations if item.get("text")],
                        family=family,
                        proxy_feedback=[item["trace"]["external_proxy"] for item in candidate_evaluations],
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "boundary_adapter"
                elif loop_decision.action == CandidateAction.CONTRAST_BOUNDARY:
                    progress(83, "Running V3 contrast boundary")
                    new_text = _generate_contrast_boundary_candidate(
                        original_text=original_text,
                        failed_candidate=str(source_item.get("text") or candidate_text),
                        family=family,
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "contrast_boundary"
                elif loop_decision.action == CandidateAction.PLAIN_REASONING:
                    progress(84, "Running V3 plain reasoning broad prose")
                    new_text = _generate_plain_reasoning_candidate(
                        original_text=original_text,
                        failed_candidates=[str(item.get("text") or "") for item in candidate_evaluations if item.get("text")],
                        family=family,
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                        rewrite_target_profile=scan_contract.rewrite_target_profile,
                        predictability_briefs=scan_contract.predictability_briefs,
                        central_judgment_plan=central_judgment_plan,
                    )
                    mode = "plain_reasoning_broad_prose"
                elif loop_decision.action == CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR:
                    progress(82, "Running V3 scanner-controlled span repair")
                    new_text, loop_target_execution = _generate_scanner_controlled_candidate(
                        original_text=str(source_item.get("text") or candidate_text or original_text),
                        original_report=source_item.get("report") or original_report,
                        scan_contract=scan_contract,
                        content_mode=content_mode,
                        family=family,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "scanner_controlled_span_repair"
                    loop_target_execution = _annotate_target_execution(
                        loop_target_execution,
                        executed_strategy="scanner_controlled_span_repair",
                        executor_engine="scanner_controlled_executor",
                    )
                elif loop_decision.action == CandidateAction.CLAIM_OWNERSHIP_REPAIR:
                    progress(84, "Running V3 claim ownership repair")
                    new_text, loop_target_execution = _generate_scanner_controlled_candidate(
                        original_text=str(source_item.get("text") or candidate_text or original_text),
                        original_report=source_item.get("report") or original_report,
                        scan_contract=scan_contract,
                        content_mode=content_mode,
                        family=family,
                        ownership_repair_mode=True,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "claim_ownership_repair"
                    loop_target_execution = _annotate_target_execution(
                        loop_target_execution,
                        executed_strategy="claim_ownership_repair",
                        executor_engine="scanner_controlled_executor",
                    )
                elif loop_decision.action == CandidateAction.TARGET_EXECUTOR:
                    if planned_problem_strategy in PROBLEM_SCANNER_CONTROLLED_STRATEGIES:
                        progress(82, f"Executing V3 planned strategy: {planned_problem_strategy}")
                        new_text, loop_target_execution = _generate_scanner_controlled_candidate(
                            original_text=str(source_item.get("text") or candidate_text or original_text),
                            original_report=source_item.get("report") or original_report,
                            scan_contract=scan_contract,
                            content_mode=content_mode,
                            family=family,
                            strategy_id=planned_problem_strategy,
                            api_key=api_key,
                            model=model,
                            base_url=base_url,
                        )
                        mode = planned_problem_strategy
                        loop_target_execution = _annotate_target_execution(
                            loop_target_execution,
                            executed_strategy=planned_problem_strategy,
                            executor_engine="scanner_controlled_executor",
                        )
                    else:
                        progress(82, "Executing V3 scanner target profile")
                        new_text, loop_target_execution = _generate_target_executor_candidate(
                            original_text=original_text,
                            scan_contract=scan_contract,
                            content_mode=content_mode,
                            family=planned_problem_strategy or family,
                            api_key=api_key,
                            model=model,
                            base_url=base_url,
                        )
                        mode = planned_problem_strategy or "target_executor"
                elif loop_decision.action == CandidateAction.REPAIR_ASSISTED_FOOTPRINT:
                    progress(83, "Repairing V3 assisted footprint paragraphs")
                    new_text, loop_target_execution = _generate_assisted_footprint_candidate(
                        original_text=original_text,
                        original_report=original_report,
                        scan_contract=scan_contract,
                        content_mode=content_mode,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "assisted_footprint_repair"
                elif loop_decision.action == CandidateAction.REPAIR_AUTHORSHIP_WINDOWS:
                    progress(85, "Repairing V3 authorship windows")
                    new_text = _generate_authorship_window_repair_candidate(
                        candidate_text=str(source_item.get("text") or candidate_text),
                        candidate_trace=source_item["trace"],
                        family=family,
                        contract=contract,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "authorship_window_repair"
                elif loop_decision.action == CandidateAction.FUSE_DETECTOR_AND_OWNERSHIP:
                    progress(86, "Fusing V3 detector movement with ownership")
                    new_text, loop_target_execution = _generate_detector_ownership_fusion_candidate(
                        source_text=original_text,
                        candidate_evaluations=candidate_evaluations,
                        family=family,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "fuse_detector_and_ownership"
                else:
                    progress(80, "Running V3 targeted repair")
                    new_text = _generate_recovery_candidate(
                        original_text=original_text,
                        failed_candidate=str(source_item.get("text") or candidate_text),
                        family=family,
                        proxy_feedback={
                            "loop_decision": loop_decision.to_dict(),
                            "external_proxy": source_item["trace"]["external_proxy"],
                        },
                        compression_policy=compression_policy,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "targeted_repair"
                source_text_for_noop = str(source_item.get("text") or "")
                if new_text and source_text_for_noop and new_text.strip() == source_text_for_noop.strip():
                    loop_target_execution = {
                        **(loop_target_execution or {}),
                        "target_execution_attempted": bool(loop_target_execution),
                        "error": "generation_failed_no_effect",
                        "strategy_stop_reason": "no_effect_candidate_discarded",
                    }
                    new_text = ""
                candidate_evaluations.append(assess_candidate(
                    text=new_text,
                    report=None,
                    mode=mode,
                    cost=len(candidate_evaluations) + 1,
                    target_execution_info=loop_target_execution,
                ))
            except Exception as exc:
                error_target_execution = None
                if loop_decision.action in {
                    CandidateAction.TARGET_EXECUTOR,
                    CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR,
                    CandidateAction.CLAIM_OWNERSHIP_REPAIR,
                }:
                    error_target_execution = target_execution_trace(
                        attempted=True,
                        target_groups=group_rewrite_targets(
                            original_text=original_text,
                            rewrite_target_profile=scan_contract.rewrite_target_profile,
                            max_groups=_max_target_executor_groups(),
                        ),
                        error=str(exc),
                    )
                    if loop_decision.action in {
                        CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR,
                        CandidateAction.CLAIM_OWNERSHIP_REPAIR,
                    }:
                        error_target_execution = {
                            **error_target_execution,
                            "scanner_controlled": True,
                            "executed_strategy": loop_decision.action.value,
                            "executor_engine": "scanner_controlled_executor",
                        }
                elif loop_decision.action == CandidateAction.REPAIR_ASSISTED_FOOTPRINT:
                    error_target_execution = target_execution_trace(
                        attempted=True,
                        target_groups=group_assisted_footprint_windows(
                            original_text=original_text,
                            authorship_window_profile=_authorship_window_profile(original_report),
                            rewrite_target_profile=scan_contract.rewrite_target_profile,
                            max_groups=_max_assisted_footprint_groups(),
                        ),
                        error=str(exc),
                    )
                candidate_evaluations.append(assess_candidate(
                    text="",
                    report=original_report,
                    mode=loop_decision.action.value,
                    cost=len(candidate_evaluations) + 1,
                    error=str(exc),
                    target_execution_info=error_target_execution,
                ))

    selected_index, selected_action, selected_reason = select_candidate_index(candidate_evaluations)
    portfolio_scores: list[dict[str, Any]] = []
    if selected_action == CandidateAction.RETURN_BEST_FOR_REVIEW:
        selected_index, portfolio_scores = select_portfolio_candidate(candidate_evaluations, family=family)
    selected = candidate_evaluations[selected_index]
    selected_trace_for_status = selected.get("trace") if isinstance(selected.get("trace"), dict) else {}
    selected_outcome_for_status = str(selected_trace_for_status.get("candidate_outcome") or "")
    no_reviewable_candidate = bool(
        selected_action == CandidateAction.RETURN_BEST_FOR_REVIEW
        and (
            not str(selected.get("text") or "").strip()
            or selected_outcome_for_status.startswith("generation_failed")
        )
    )
    if no_reviewable_candidate:
        public_status = RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value
    else:
        public_status = {
            CandidateAction.ACCEPT_STRICT: RewriteGoalStatus.AI_MITIGATED.value,
            CandidateAction.ACCEPT_EXTERNAL: "rewrite_candidate_generated_needs_external_review",
            CandidateAction.RETURN_BEST_FOR_REVIEW: "rewrite_candidate_generated_needs_external_review",
        }[selected_action]
    converged = selected_action == CandidateAction.ACCEPT_STRICT
    convergence_reason = (
        "rewrite_v3_no_safe_candidate_generated"
        if no_reviewable_candidate
        else {
            CandidateAction.ACCEPT_STRICT: "rewrite_v3_strict_goal_met",
            CandidateAction.ACCEPT_EXTERNAL: "rewrite_v3_external_calibrated_candidate_requires_review",
            CandidateAction.RETURN_BEST_FOR_REVIEW: "rewrite_v3_best_candidate_needs_external_review",
        }[selected_action]
    )
    final_text = selected["text"] or original_text
    final_report = selected["report"] or original_report

    elapsed = time.time() - started
    candidate_trace = [item["trace"] for item in candidate_evaluations]
    strategy_stack = [step.to_dict() for step in getattr(strategy_plan, "steps", ()) or ()]
    executed_problem_groups = []
    problem_group_results = []
    for trace in candidate_trace:
        target_trace = trace.get("target_execution_trace") if isinstance(trace.get("target_execution_trace"), dict) else {}
        executed_problem_groups.extend(
            group.get("group_id")
            for group in target_trace.get("target_groups") or []
            if isinstance(group, dict) and group.get("group_id")
        )
        problem_group_results.append({
            "generation_mode": trace.get("generation_mode"),
            "detector_movement": trace.get("detector_movement"),
            "target_gate_passed": trace.get("target_gate_passed"),
            "candidate_outcome": trace.get("candidate_outcome"),
            "footprint_delta": trace.get("footprint_delta"),
            "target_movement": trace.get("target_movement"),
        })
    final_goal = selected["trace"]["goal"]
    selected_trace = selected["trace"]
    best_candidate_external_review_required = public_status == "rewrite_candidate_generated_needs_external_review"
    public_candidate_warning = ""
    if best_candidate_external_review_required:
        public_candidate_warning = "best_candidate_requires_external_review"
        if selected_trace.get("candidate_outcome") == "corrupted_output":
            public_candidate_warning = "best_candidate_failed_text_integrity"
        elif not selected_trace.get("detector_movement"):
            public_candidate_warning = "best_candidate_has_no_detector_footprint_movement"
        elif not selected_trace.get("target_gate_passed", True):
            public_candidate_warning = "best_candidate_has_no_rewrite_target_movement"
        elif selected_trace.get("validity_status") != "valid":
            public_candidate_warning = "best_candidate_is_invalid_but_detector_improved"
    if public_status != RewriteGoalStatus.AI_MITIGATED.value:
        final_goal = {
            **final_goal,
            "status": (
                RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value
                if public_status != "rewrite_candidate_generated_needs_external_review"
                else public_status
            ),
            "goal_met": False,
            "reason": convergence_reason,
        }
    summary = {
        "rewrite_pipeline_version": "rewrite_v3_external_calibrated",
        "outcome": public_status,
        "public_status": public_status,
        "public_candidate_warning": public_candidate_warning,
        "best_candidate_external_review_required": best_candidate_external_review_required,
        "strict_goal_status": final_goal.get("status"),
        "rewrite_goal_status": final_goal,
        "reference_ai": reference_ai,
        "required_ai_drop": required_ai_drop,
        "target_ai_score": target_ai_score,
        "content_router_trace": content_router_trace,
        "scan_contract": scan_contract.to_dict(),
        "v3_route": v3_route.to_dict(),
        "v3_strategy_plan": strategy_plan.to_dict(),
        "problem_inventory_before": original_problem_inventory,
        "strategy_stack": strategy_stack,
        "executed_problem_groups": list(dict.fromkeys(str(item) for item in executed_problem_groups if item)),
        "problem_group_results": problem_group_results,
        "unresolved_problem_groups": selected_trace.get("unresolved_problem_groups") if isinstance(selected_trace, dict) else [],
        "strategy_switch_reason": loop_trace[-1]["reason"] if loop_trace else "",
        "strategy_trace": [{
            "strategy_family": family,
            "first_strategy_step": first_strategy_step,
            "strategy_stack": strategy_stack,
            "problem_inventory_driven": problem_inventory_driven,
            "external_calibrated": True,
            "generation_mode": generation_mode,
            "executed_initial_strategy": generation_mode,
            "executor_engine": target_execution.get("executor_engine") if isinstance(target_execution, dict) else "",
            "first_strategy_obeyed": bool(
                not problem_inventory_driven
                or not first_strategy_step
                or generation_mode == first_strategy_step
                or first_strategy_step == "unit_preserving_prune_bridge"
            ),
            "scanner_controlled_executor_available": _scanner_controlled_executor_available(scan_contract),
            "scanner_controlled_executor_first": bool(
                generation_mode == "scanner_controlled_executor"
                or (
                    _uses_problem_scanner_controlled_strategy(generation_mode)
                    and isinstance(target_execution, dict)
                    and target_execution.get("executor_engine") == "scanner_controlled_executor"
                )
            ),
            "scanner_controlled_config": _scanner_controlled_config().to_dict(),
            "target_executor_available": _target_executor_available(scan_contract),
            "prune_bridge_available": _prune_bridge_available(scan_contract),
            "prune_bridge_first": generation_mode == "unit_preserving_prune_bridge",
            "target_executor_first": generation_mode == "target_executor",
            "problem_target_executor_first": _uses_problem_target_executor_strategy(generation_mode),
            "compression_policy": compression_policy.to_dict(),
            "single_shot_word_limit": _single_shot_word_limit(),
            "chunk_word_limit": _chunk_word_limit(),
            "force_unit_chunks": force_unit_chunks,
            "central_judgment_plan": central_judgment_plan,
        }],
        "candidate_trace": candidate_trace,
        "candidate_loop_trace": loop_trace,
        "portfolio_scores": portfolio_scores,
        "selected_candidate": candidate_trace[selected_index],
        "diagnostic_candidate_text": (
            final_text
            if os.environ.get("DRAFTPROOF_REWRITE_V3_EXPOSE_DIAGNOSTIC_TEXT", "0").lower() in {"1", "true", "yes"}
            else None
        ),
        "stage_timings": [{
            "stage": "rewrite_v3_external_calibrated",
            "seconds": round(elapsed, 3),
            "selected": public_status != RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE.value,
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
    md_path = out_dir / f"draftproof_rewrite_v3_{ts}.md"
    pdf_path = out_dir / f"draftproof_rewrite_v3_{ts}.pdf"
    json_path = out_dir / f"draftproof_rewrite_v3_{ts}.json"
    md_text = render_rewrite_report(summary=summary, sentence_comparison=sentence_comparison, ai_findings=[], verbose=False)
    md_path.write_text(md_text, encoding="utf-8")
    render_pdf(md_text, str(pdf_path))
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    progress(88, "External-calibrated rewrite V3 complete")
    return {
        "status": public_status,
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "json_path": str(json_path),
        "result": result_obj,
        "elapsed": elapsed,
    }
