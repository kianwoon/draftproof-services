"""Strict rewrite goal contract for the scan-driven V2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import os
import re
from typing import Any

from rewrite_controller.eligible_span_density import build_eligible_span_density_contract
from rewrite_pipeline_core.gates.ai_footprint import _ai_footprint_gate_status
from rewrite_pipeline_core.scoring.profiles import _turnitin_like_ai_gate_status

from .external_calibration import (
    calibration_policy_to_dict,
    load_external_calibration_labels,
    summarize_external_calibration_records,
)


class RewriteGoalStatus(str, Enum):
    AI_MITIGATED = "ai_mitigated"
    MITIGATION_FAILED_NO_SAFE_CANDIDATE = "mitigation_failed_no_safe_candidate"
    NEEDS_AUTHOR_CONTEXT = "needs_author_context"
    ORIGINAL_PRESERVED = "original_preserved"


@dataclass(frozen=True)
class RewriteGoalEvaluation:
    status: RewriteGoalStatus
    goal_met: bool
    detector_safe: bool
    strict_ai_safe_band_achieved: bool
    turnitin_like_target_met: bool
    eligible_span_density_safe: bool
    reason: str
    ai_footprint_gate: dict[str, Any]
    turnitin_like_gate: dict[str, Any]
    eligible_span_density_gate: dict[str, Any]
    external_detector_proxy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["version"] = "rewrite_goal_contract_v2"
        return payload


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if item.strip()]


def _paragraphs(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"\n\s*\n+", str(text or "").strip()) if item.strip()]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text or "")))


def _coefficient_of_variation(values: list[int | float]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float)) and float(value) > 0.0]
    if len(numeric) < 2:
        return 0.0
    mean = sum(numeric) / len(numeric)
    if mean <= 0.0:
        return 0.0
    variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    return (variance ** 0.5) / mean


def _flat_footprint_after(footprint: dict[str, Any]) -> dict[str, float]:
    after = footprint.get("after") if isinstance(footprint.get("after"), dict) else {}
    flat: dict[str, float] = {}
    for bucket in (
        "authorship_footprint",
        "structural_footprint",
        "semantic_footprint",
        "grounding_footprint",
    ):
        values = after.get(bucket) if isinstance(after.get(bucket), dict) else {}
        for key, value in values.items():
            flat[key] = _num(value)
    flat["external_ai_flag_risk"] = _num(after.get("external_ai_flag_risk"))
    return flat


def _external_detector_proxy_status(
    *,
    candidate_text: str,
    ai_footprint_gate: dict[str, Any],
    turnitin_like_gate: dict[str, Any],
    eligible_span_density_gate: dict[str, Any],
) -> dict[str, Any]:
    """V2-only external-detector calibration proxy.

    This does not replace the scanner score. It is a stricter acceptance proxy
    for rewrite candidates, aimed at the external-detector failure mode where a
    polished candidate passes internal formula/density gates but still reads as
    generic, smooth, or highly predictable.
    """
    after = _flat_footprint_after(ai_footprint_gate)
    turnitin_after = turnitin_like_gate.get("after") if isinstance(turnitin_like_gate.get("after"), dict) else {}
    turnitin_components = turnitin_after.get("components") if isinstance(turnitin_after.get("components"), dict) else {}
    sentences = _sentences(candidate_text)
    sentence_lengths = [_word_count(sentence) for sentence in sentences]
    sentence_cv = _coefficient_of_variation(sentence_lengths)
    paragraphs = _paragraphs(candidate_text)
    paragraph_lengths = [_word_count(paragraph) for paragraph in paragraphs]
    paragraph_cv = _coefficient_of_variation(paragraph_lengths)
    density_ratio = _num(eligible_span_density_gate.get("unsafe_eligible_word_ratio"))
    human_anchor = _num(eligible_span_density_gate.get("human_anchor_score"))

    low_sentence_variation_penalty = 0.0
    if len(sentence_lengths) >= 6 and sentence_cv < 0.25:
        low_sentence_variation_penalty = 18.0
    elif len(sentence_lengths) >= 6 and sentence_cv < 0.35:
        low_sentence_variation_penalty = 12.0
    elif len(sentence_lengths) >= 6 and sentence_cv < 0.45:
        low_sentence_variation_penalty = 6.0

    paragraph_symmetry_penalty = 0.0
    if len(paragraph_lengths) >= 4 and paragraph_cv < 0.18:
        paragraph_symmetry_penalty = 10.0
    elif len(paragraph_lengths) >= 4 and paragraph_cv < 0.28:
        paragraph_symmetry_penalty = 6.0

    weak_anchor_penalty = 5.0 if human_anchor and human_anchor < 30.0 else 0.0
    generic_source_compound_penalty = (
        8.0
        if after.get("generic_assertion_risk", 0.0) >= 85.0
        and after.get("source_grounding_risk", 0.0) >= 85.0
        and after.get("topk_calibrated_risk", 0.0) > 25.0
        else 0.0
    )

    weighted_risk = (
        after.get("ai_likelihood", 0.0) * 0.20
        + after.get("topk_calibrated_risk", 0.0) * 0.18
        + after.get("semantic_uniformity", 0.0) * 0.10
        + after.get("generic_assertion_risk", 0.0) * 0.09
        + after.get("unsupported_claim_risk", 0.0) * 0.04
        + after.get("source_grounding_risk", 0.0) * 0.025
        + density_ratio * 0.12
        + after.get("discourse_regularity", 0.0) * 0.05
        + _num(turnitin_components.get("patchwork_expansion")) * 0.06
        + low_sentence_variation_penalty
        + paragraph_symmetry_penalty
        + weak_anchor_penalty
        + generic_source_compound_penalty
    )
    score = round(min(100.0, max(0.0, weighted_risk)), 3)
    calibration_policy = calibration_policy_to_dict()
    threshold_summary = calibration_policy.get("threshold_summary") if isinstance(calibration_policy.get("threshold_summary"), dict) else {}
    calibrated_safe_threshold = threshold_summary.get("safe_threshold")
    safe_threshold = (
        float(calibrated_safe_threshold)
        if isinstance(calibrated_safe_threshold, (int, float))
        and threshold_summary.get("status") == "derived_from_labeled_proxy_records"
        and os.environ.get("DRAFTPROOF_REWRITE_V2_USE_CALIBRATED_PROXY_THRESHOLDS", "1").lower() not in {"0", "false", "no"}
        else _float_env("DRAFTPROOF_REWRITE_V2_EXTERNAL_PROXY_SAFE_MAX", 38.0)
    )
    warning_threshold = _float_env("DRAFTPROOF_REWRITE_V2_EXTERNAL_PROXY_WARN_MAX", 32.0)
    hard_blockers = []
    if after.get("topk_calibrated_risk", 0.0) > _float_env("DRAFTPROOF_REWRITE_V2_EXTERNAL_PROXY_TOPK_MAX", 25.0):
        hard_blockers.append("topk_calibrated_risk_above_external_safe_band")
    if density_ratio > _float_env("DRAFTPROOF_REWRITE_V2_EXTERNAL_PROXY_DENSITY_MAX", 12.0):
        hard_blockers.append("eligible_span_density_above_external_safe_band")
    if low_sentence_variation_penalty >= 12.0 and after.get("ai_likelihood", 0.0) >= 30.0:
        hard_blockers.append("low_sentence_variation_with_active_ai_likelihood")
    safe = bool(score <= safe_threshold and not hard_blockers)
    if safe:
        outcome = "external_proxy_safe"
    elif score <= warning_threshold and hard_blockers:
        outcome = "external_proxy_blocked_by_hard_signal"
    else:
        outcome = "external_proxy_risk_high"
    return {
        "version": "external_detector_risk_proxy_v2",
        "safe": safe,
        "score": score,
        "safe_threshold": safe_threshold,
        "warning_threshold": warning_threshold,
        "outcome": outcome,
        "hard_blockers": hard_blockers,
        "signals": {
            "ai_likelihood": round(after.get("ai_likelihood", 0.0), 3),
            "topk_calibrated_risk": round(after.get("topk_calibrated_risk", 0.0), 3),
            "semantic_uniformity": round(after.get("semantic_uniformity", 0.0), 3),
            "generic_assertion_risk": round(after.get("generic_assertion_risk", 0.0), 3),
            "unsupported_claim_risk": round(after.get("unsupported_claim_risk", 0.0), 3),
            "source_grounding_risk": round(after.get("source_grounding_risk", 0.0), 3),
            "unsafe_eligible_word_ratio": round(density_ratio, 3),
            "patchwork_expansion": round(_num(turnitin_components.get("patchwork_expansion")), 3),
            "sentence_length_cv": round(sentence_cv, 3),
            "paragraph_length_cv": round(paragraph_cv, 3),
            "human_anchor_score": round(human_anchor, 3),
        },
        "penalties": {
            "low_sentence_variation": low_sentence_variation_penalty,
            "paragraph_symmetry": paragraph_symmetry_penalty,
            "weak_human_anchor": weak_anchor_penalty,
            "generic_source_compound": generic_source_compound_penalty,
        },
        "calibration_policy": calibration_policy,
        "calibration_summary": summarize_external_calibration_records(load_external_calibration_labels()),
    }


def _finding_total(report: dict | None) -> int:
    findings = (report or {}).get("findings", {}) if isinstance(report, dict) else {}
    if isinstance(findings, int):
        return max(0, findings)
    if not isinstance(findings, dict):
        return 0
    return sum(len(findings.get(tier, [])) for tier in ("critical", "high", "medium", "low"))


def _review_burden(report: dict | None) -> int:
    findings = (report or {}).get("findings", {}) if isinstance(report, dict) else {}
    if isinstance(findings, int):
        return max(0, findings)
    if not isinstance(findings, dict):
        return 0
    return sum(len(findings.get(tier, [])) for tier in ("critical", "high", "medium"))


def _weighted_severity(report: dict | None) -> int:
    findings = (report or {}).get("findings", {}) if isinstance(report, dict) else {}
    if isinstance(findings, int):
        return max(0, findings)
    if not isinstance(findings, dict):
        return 0
    weights = {"critical": 8, "high": 5, "medium": 2, "low": 1}
    return sum(len(findings.get(tier, [])) * weights[tier] for tier in weights)


def needs_author_context(scan_report: dict | None) -> bool:
    if not isinstance(scan_report, dict):
        return False
    explicit_keys = {
        "needs_author_context",
        "requires_author_context",
        "author_context_required",
        "missing_author_context",
    }
    stack = [scan_report]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                key_text = str(key).lower()
                if key_text in explicit_keys and (
                    value is True
                    or (isinstance(value, str) and value.strip().lower() in {"true", "yes", "required", "blocked"})
                ):
                    return True
                if key_text in {"status", "outcome", "stop_reason", "blocker", "reason"}:
                    text = str(value).lower()
                    if any(token in text for token in (
                        "needs_author_context",
                        "requires_author_context",
                        "author_context_required",
                        "missing_author_context",
                    )):
                        return True
                if isinstance(value, (dict, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, (dict, list)))
    # Legacy free-text fallback is deliberately narrow. General guidance such as
    # "author-owned evidence additions" must not block automated mitigation.
    intelligence = scan_report.get("scan_intelligence") or {}
    mitigation = (
        intelligence.get("mitigation_inputs")
        if isinstance(intelligence.get("mitigation_inputs"), dict)
        else {}
    )
    for payload in (
        scan_report.get("ai_mitigation"),
        mitigation.get("ai_mitigation_plan"),
        scan_report.get("generation_handoff"),
        intelligence.get("generation_handoff"),
    ):
        if not isinstance(payload, dict):
            continue
        text = " ".join(str(value).lower() for value in payload.values())
        if any(phrase in text for phrase in (
            "needs author context",
            "requires author context",
            "author context required",
            "missing author context",
        )):
            return True
    return False


def evaluate_rewrite_goal(
    *,
    original_text: str,
    candidate_text: str,
    original_report: dict | None,
    candidate_report: dict | None,
    no_text_change: bool = False,
) -> RewriteGoalEvaluation:
    review_delta = _review_burden(candidate_report) - _review_burden(original_report)
    severity_delta = _weighted_severity(candidate_report) - _weighted_severity(original_report)
    candidate_findings = (candidate_report or {}).get("findings") if isinstance(candidate_report, dict) else {}
    original_findings = (original_report or {}).get("findings") if isinstance(original_report, dict) else {}
    candidate_findings = candidate_findings if isinstance(candidate_findings, dict) else {}
    original_findings = original_findings if isinstance(original_findings, dict) else {}
    critical_high_delta = (
        sum(len(candidate_findings.get(tier, [])) for tier in ("critical", "high"))
        - sum(len(original_findings.get(tier, [])) for tier in ("critical", "high"))
    )
    footprint = _ai_footprint_gate_status(
        original_report,
        candidate_report,
        review_burden_delta=review_delta,
        weighted_severity_delta=severity_delta,
        critical_high_delta=critical_high_delta,
    )
    turnitin = _turnitin_like_ai_gate_status(
        original_report,
        candidate_report,
        review_burden_delta=review_delta,
        weighted_severity_delta=severity_delta,
        critical_high_delta=critical_high_delta,
    )
    density = build_eligible_span_density_contract(candidate_text, candidate_report)
    external_proxy = _external_detector_proxy_status(
        candidate_text=candidate_text,
        ai_footprint_gate=footprint,
        turnitin_like_gate=turnitin,
        eligible_span_density_gate=density,
    )
    strict_safe = bool(footprint.get("safe_band"))
    turnitin_target = bool(turnitin.get("target_met") or turnitin.get("safe_band"))
    density_safe = bool(density.get("safe"))
    external_safe = bool(external_proxy.get("safe"))
    detector_safe = bool(strict_safe and turnitin_target and density_safe and external_safe)
    if no_text_change:
        status = RewriteGoalStatus.ORIGINAL_PRESERVED
        reason = "no_safe_rewrite_applied"
    elif detector_safe:
        status = RewriteGoalStatus.AI_MITIGATED
        reason = "strict_detector_safe_goal_met"
    elif needs_author_context(original_report):
        status = RewriteGoalStatus.NEEDS_AUTHOR_CONTEXT
        reason = "scan_requires_author_context"
    else:
        status = RewriteGoalStatus.MITIGATION_FAILED_NO_SAFE_CANDIDATE
        reason = "candidate_failed_strict_detector_safe_goal"
    return RewriteGoalEvaluation(
        status=status,
        goal_met=detector_safe,
        detector_safe=detector_safe,
        strict_ai_safe_band_achieved=strict_safe,
        turnitin_like_target_met=turnitin_target,
        eligible_span_density_safe=density_safe,
        reason=reason,
        ai_footprint_gate=footprint,
        turnitin_like_gate=turnitin,
        eligible_span_density_gate=density,
        external_detector_proxy=external_proxy,
    )
