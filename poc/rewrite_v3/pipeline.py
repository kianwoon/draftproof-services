"""External-calibrated rewrite pipeline V3 entrypoint."""

from __future__ import annotations

import json
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
from .candidate_loop import CandidateAction, decide_next_action, select_candidate_index
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
from .router import route_from_scan_contract
from .scanner_contract import RewriteRiskClass, ScanContract, build_scan_contract
from .style_library import examples_for_family
from .strategy_plan import build_strategy_plan
from .target_executor import (
    SUPPORTED_TARGET_OPERATIONS,
    apply_target_replacements,
    batch_target_groups,
    build_target_executor_prompt,
    group_rewrite_targets,
    parse_target_replacements,
    target_execution_trace,
)


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
    moved = (
        risk_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_FOOTPRINT_RISK_DROP", 2.0)
        or fraction_ai_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_FRACTION_AI_DROP", 0.03)
        or assisted_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_ASSISTED_FRACTION_DROP", 0.04)
        or risky_density_drop >= _float_env("DRAFTPROOF_REWRITE_V3_MIN_RISKY_DENSITY_DROP", 0.04)
        or high_conf_drop >= 1.0
    )
    return {
        "before_risk": before_risk,
        "after_risk": after_risk,
        "risk_drop": round(risk_drop, 3),
        "fraction_ai_drop": round(fraction_ai_drop, 4),
        "fraction_ai_assisted_drop": round(assisted_drop, 4),
        "risky_window_density_drop": round(risky_density_drop, 4),
        "high_confidence_risky_window_drop": round(high_conf_drop, 3),
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


def _text_integrity(source_text: str, candidate_text: str) -> dict[str, Any]:
    source = str(source_text or "")
    candidate = str(candidate_text or "")

    def metrics(text: str) -> dict[str, Any]:
        chars = list(text)
        char_count = max(1, len(chars))
        tokens = text.split()
        alpha_run = 0
        max_alpha_run = 0
        punctuation = 0
        non_ascii_punctuation = 0
        for char in chars:
            if char.isalpha():
                alpha_run += 1
                max_alpha_run = max(max_alpha_run, alpha_run)
            else:
                alpha_run = 0
            category = unicodedata.category(char)
            if category.startswith("P"):
                punctuation += 1
                if ord(char) > 127:
                    non_ascii_punctuation += 1
        long_tokens = [token for token in tokens if len(token) >= 24]
        return {
            "char_count": len(chars),
            "token_count": len(tokens),
            "space_ratio": sum(1 for char in chars if char.isspace()) / char_count,
            "punctuation_ratio": punctuation / char_count,
            "non_ascii_punctuation_ratio": non_ascii_punctuation / char_count,
            "max_alpha_run": max_alpha_run,
            "long_token_ratio": len(long_tokens) / max(1, len(tokens)),
        }

    src = metrics(source)
    cand = metrics(candidate)
    failures: list[str] = []
    if candidate.strip() and cand["space_ratio"] < max(0.02, src["space_ratio"] * 0.55):
        failures.append("spacing_collapse")
    if cand["max_alpha_run"] >= max(36, src["max_alpha_run"] * 2):
        failures.append("merged_word_run")
    if cand["long_token_ratio"] > max(0.04, src["long_token_ratio"] + 0.035):
        failures.append("merged_long_tokens")
    if cand["non_ascii_punctuation_ratio"] > max(0.025, src["non_ascii_punctuation_ratio"] + 0.02):
        failures.append("punctuation_script_shift")
    if cand["punctuation_ratio"] > max(0.18, src["punctuation_ratio"] + 0.10):
        failures.append("punctuation_density_shift")
    return {
        "passed": not failures,
        "failures": failures,
        "source": src,
        "candidate": cand,
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
    return max(1200, min(12000, int(words * 2.4) + 900))


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
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
    return LLMGateway(LLMConfig(
        api_key=api_key,
        model=model or os.environ.get("LLM_MODEL") or "deepseek/deepseek-chat",
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
            central_judgment_plan=central_judgment_plan,
        )
    elif family == "clean_texture_boundary":
        prompt = build_clean_texture_boundary_prompt(
            original_text=original_text,
            scan_report=original_report,
            style_examples=examples,
            rewrite_target_profile=rewrite_target_profile,
            central_judgment_plan=central_judgment_plan,
        )
    else:
        prompt = build_document_rhythm_prompt(
            original_text=original_text,
            compression_policy=compression_policy,
            style_examples=examples,
            rewrite_target_profile=rewrite_target_profile,
            central_judgment_plan=central_judgment_plan,
        )
    token_words = word_count(original_text) if family == "clean_texture_boundary" else compression_policy.max_words
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(token_words))
    kwargs = _clean_texture_chat_kwargs() if family == "clean_texture_boundary" else {}
    return clean_v3_candidate_output(gateway.chat(
        prompt,
        system="Return only the rewritten document as plain text.",
        **kwargs,
    ).content)


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
    gateway = _gateway(api_key, model, base_url, max_tokens=_max_tokens_for_words(compression_policy.max_words))
    return clean_v3_candidate_output(gateway.chat(prompt, system="Return only the rewritten document as plain text.").content)


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
    central_judgment_plan: dict[str, Any] | None = None,
) -> str:
    prompt = build_plain_reasoning_broad_prose_prompt(
        original_text=original_text,
        failed_candidates=failed_candidates,
        compression_policy=compression_policy,
        style_examples=examples_for_family(family),
        rewrite_target_profile=rewrite_target_profile,
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


def _strategy_plan_prefers_target_executor(strategy_plan: Any, scan_contract: ScanContract) -> bool:
    if not _target_executor_available(scan_contract):
        return False
    target_step_ids = {
        "protected_section_rewrite",
        "citation_anchor_guard",
        "authorship_window_repair",
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
    replacements: list[dict[str, str]] = []
    batch_trace: list[dict[str, Any]] = []
    batch_errors: list[str] = []
    for batch_index, batch in enumerate(batch_target_groups(groups, batch_size=_target_executor_batch_size()), start=1):
        prompt = build_target_executor_prompt(
            target_groups=batch,
            content_mode=content_mode,
            strategy_family=family,
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
                central_judgment_plan=central_judgment_plan,
            )
        elif family == "clean_texture_boundary":
            prompt = build_clean_texture_boundary_chunk_prompt(
                source_units=chunk,
                global_plan=global_plan,
                style_examples=examples,
                rewrite_target_profile=rewrite_target_profile,
                central_judgment_plan=central_judgment_plan,
            )
        else:
            prompt = build_document_rhythm_chunk_prompt(
                source_units=chunk,
                global_plan=global_plan,
                compression_policy=chunk_policy,
                style_examples=examples,
                rewrite_target_profile=rewrite_target_profile,
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

    if replay_candidate_records:
        candidate_text, candidate_report = _candidate_from_replay(replay_candidate_records[0], original_report)
    elif full_rewrite_allowed:
        try:
            if _strategy_plan_prefers_target_executor(strategy_plan, scan_contract):
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

    reference_ai = _badge_ai(original_report)
    target_ai_score = float(reference_ai) - float(required_ai_drop) if isinstance(reference_ai, (int, float)) else None
    expected_unit_count = len(source_generation_units)
    original_footprint = _ai_footprint_profile(original_report) or scan_contract.ai_footprint_profile
    original_target_profile = _rewrite_target_profile(original_report) or scan_contract.rewrite_target_profile

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
        should_scan_candidate = bool(text)
        scanned_report = report
        if should_scan_candidate and scanned_report is None:
            progress(78, f"Scanning V3 {mode} candidate")
            scanned_report = _scan_report(text)
        elif scanned_report is None:
            scanned_report = original_report
        candidate_footprint = _ai_footprint_profile(scanned_report)
        footprint_delta = _footprint_delta(original_footprint, candidate_footprint)
        candidate_target_profile = _rewrite_target_profile(scanned_report)
        target_movement = _target_profile_movement(original_target_profile, candidate_target_profile)
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
                and target_movement["moved"]
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
        can_select = bool(
            text
            and validity == "valid"
            and footprint_delta["moved"]
            and target_movement["moved"]
            and integrity_result["passed"]
        )
        target_trace = target_execution_info if isinstance(target_execution_info, dict) else {}
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
                "authorship_window_profile": _authorship_window_profile(scanned_report),
                "footprint_before": original_footprint,
                "footprint_after": candidate_footprint,
                "footprint_delta": footprint_delta,
                "target_profile_before": original_target_profile,
                "target_profile_after": candidate_target_profile,
                "target_movement": target_movement,
                "target_gate_passed": bool(target_movement["moved"]),
                "target_execution_available": _target_executor_available(scan_contract),
                "assisted_footprint_executor_available": _assisted_footprint_executor_available(scan_contract),
                "target_execution_attempted": target_attempted,
                "target_execution_trace": target_trace,
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
                        ("central_profile_not_improved", _number(central_profile_score.get("weighted_delta")) <= 0.0),
                        ("validation_failed", not validation_result.passed),
                        ("compression_rejected", not compression_ok),
                        ("semantic_drift", not semantic_safe),
                        ("text_integrity_failed", not integrity_result["passed"]),
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
    if generation_mode == "target_executor":
        tried_actions.add(CandidateAction.TARGET_EXECUTOR)
    if replay_candidate_records and len(replay_candidate_records) > 1:
        for index, record in enumerate(replay_candidate_records[1:], start=2):
            replay_text, replay_report = _candidate_from_replay(record, original_report)
            candidate_evaluations.append(assess_candidate(
                text=replay_text,
                report=replay_report,
                mode=f"replay_recovery_{index}",
                cost=index,
            ))
    elif full_rewrite_allowed:
        while (time.time() - started) < max_runtime_seconds and len(candidate_evaluations) < 5:
            loop_decision = decide_next_action(
                candidate_evaluations,
                has_positive_boundaries=bool(examples_for_family(family).get("positive") or []),
                tried_actions=tried_actions,
            )
            loop_trace.append(loop_decision.to_dict())
            if loop_decision.action in {
                CandidateAction.ACCEPT_STRICT,
                CandidateAction.ACCEPT_EXTERNAL,
                CandidateAction.RETURN_BEST_FOR_REVIEW,
            }:
                break
            tried_actions.add(loop_decision.action)
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
                        central_judgment_plan=central_judgment_plan,
                    )
                    mode = "plain_reasoning_broad_prose"
                elif loop_decision.action == CandidateAction.TARGET_EXECUTOR:
                    progress(82, "Executing V3 scanner target profile")
                    new_text, loop_target_execution = _generate_target_executor_candidate(
                        original_text=original_text,
                        scan_contract=scan_contract,
                        content_mode=content_mode,
                        family=family,
                        api_key=api_key,
                        model=model,
                        base_url=base_url,
                    )
                    mode = "target_executor"
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
                candidate_evaluations.append(assess_candidate(
                    text=new_text,
                    report=None,
                    mode=mode,
                    cost=len(candidate_evaluations) + 1,
                    target_execution_info=loop_target_execution,
                ))
            except Exception as exc:
                error_target_execution = None
                if loop_decision.action == CandidateAction.TARGET_EXECUTOR:
                    error_target_execution = target_execution_trace(
                        attempted=True,
                        target_groups=group_rewrite_targets(
                            original_text=original_text,
                            rewrite_target_profile=scan_contract.rewrite_target_profile,
                            max_groups=_max_target_executor_groups(),
                        ),
                        error=str(exc),
                    )
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
    public_status = {
        CandidateAction.ACCEPT_STRICT: RewriteGoalStatus.AI_MITIGATED.value,
        CandidateAction.ACCEPT_EXTERNAL: "rewrite_candidate_generated_needs_external_review",
        CandidateAction.RETURN_BEST_FOR_REVIEW: "rewrite_candidate_generated_needs_external_review",
    }[selected_action]
    converged = selected_action == CandidateAction.ACCEPT_STRICT
    convergence_reason = {
        CandidateAction.ACCEPT_STRICT: "rewrite_v3_strict_goal_met",
        CandidateAction.ACCEPT_EXTERNAL: "rewrite_v3_external_calibrated_candidate_requires_review",
        CandidateAction.RETURN_BEST_FOR_REVIEW: "rewrite_v3_best_candidate_needs_external_review",
    }[selected_action]
    final_text = selected["text"] or original_text
    final_report = selected["report"] or original_report

    elapsed = time.time() - started
    candidate_trace = [item["trace"] for item in candidate_evaluations]
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
        "strategy_trace": [{
            "strategy_family": family,
            "first_strategy_step": _first_strategy_step_id(strategy_plan),
            "external_calibrated": True,
            "generation_mode": generation_mode,
            "target_executor_available": _target_executor_available(scan_contract),
            "target_executor_first": generation_mode == "target_executor",
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
