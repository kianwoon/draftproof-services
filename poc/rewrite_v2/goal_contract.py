"""Strict rewrite goal contract for the scan-driven V2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from rewrite_controller.eligible_span_density import build_eligible_span_density_contract
from rewrite_pipeline_core.gates.ai_footprint import _ai_footprint_gate_status
from rewrite_pipeline_core.scoring.profiles import _turnitin_like_ai_gate_status


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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["version"] = "rewrite_goal_contract_v2"
        return payload


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
                if key_text in explicit_keys and bool(value):
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
    strict_safe = bool(footprint.get("safe_band"))
    turnitin_target = bool(turnitin.get("target_met") or turnitin.get("safe_band"))
    density_safe = bool(density.get("safe"))
    detector_safe = bool(strict_safe and turnitin_target and density_safe)
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
    )
