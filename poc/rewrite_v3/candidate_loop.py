"""Issue-driven candidate loop helpers for rewrite V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class CandidateIssue(str, Enum):
    VALIDATION_FAILED = "validation_failed"
    STRUCTURE_CHANGED = "structure_changed"
    ANCHOR_MISSING = "anchor_missing"
    COMPRESSION_REJECTED = "compression_rejected"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    SEMANTIC_DRIFT = "semantic_drift"
    INTERNAL_AI_BACKFIRE = "internal_ai_backfire"
    INSUFFICIENT_TOPK_DROP = "insufficient_topk_drop"
    TOPK_CANDIDATE_REJECTED = "topk_candidate_rejected"
    NO_EFFECT_SPAN_PATCH = "no_effect_span_patch"
    ZERO_CHANGE_TOPK = "zero_change_topk"
    SELF_REPORT_MISMATCH = "self_report_mismatch"
    INSUFFICIENT_SPAN_MOVEMENT = "insufficient_span_movement"
    WRITING_QUALITY_COLLAPSE = "writing_quality_collapse"
    SEGMENT_AI_FOOTPRINT = "segment_ai_footprint"
    OWNERSHIP_MISSING = "ownership_missing"
    PROXY_NOT_ACCEPTED = "proxy_not_accepted"
    NO_DETECTOR_MOVEMENT = "no_detector_movement"
    NO_TARGET_MOVEMENT = "no_target_movement"
    TEXT_CORRUPTED = "text_corrupted"
    GENERATION_FAILED = "generation_failed"


class CandidateAction(str, Enum):
    ACCEPT_STRICT = "accept_strict"
    ACCEPT_EXTERNAL = "accept_external"
    REPAIR_CONTRACT = "repair_contract"
    REPAIR_STRUCTURE = "repair_structure"
    CONTRAST_BOUNDARY = "contrast_boundary"
    PLAIN_REASONING = "plain_reasoning"
    REPAIR_AUTHORSHIP_WINDOWS = "repair_authorship_windows"
    CLAIM_OWNERSHIP_REPAIR = "claim_ownership_repair"
    SCANNER_CONTROLLED_SPAN_REPAIR = "scanner_controlled_span_repair"
    REPAIR_TARGETED = "repair_targeted"
    TARGET_EXECUTOR = "target_executor"
    REPAIR_ASSISTED_FOOTPRINT = "repair_assisted_footprint"
    ADAPT_BOUNDARY = "adapt_boundary"
    RETURN_BEST_FOR_REVIEW = "return_best_for_review"


@dataclass(frozen=True)
class LoopDecision:
    action: CandidateAction
    source_index: int
    issues: tuple[CandidateIssue, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["issues"] = [issue.value for issue in self.issues]
        return payload


def _issue_values(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple)):
        return set()
    return {str(value) for value in values}


def _topk_effect_failures(trace: dict[str, Any]) -> set[str]:
    failures: set[str] = set()
    direct = trace.get("topk_effect_failures")
    failures.update(_issue_values(direct))
    target_trace = trace.get("target_execution_trace") if isinstance(trace.get("target_execution_trace"), dict) else {}
    stage_rows = target_trace.get("prompt_stage_trace") if isinstance(target_trace.get("prompt_stage_trace"), list) else []
    for stage in stage_rows:
        if not isinstance(stage, dict):
            continue
        diagnostics = stage.get("parse_diagnostics") if isinstance(stage.get("parse_diagnostics"), dict) else {}
        effect_rows = diagnostics.get("effect_status") if isinstance(diagnostics.get("effect_status"), list) else []
        for row in effect_rows:
            if isinstance(row, dict):
                failures.update(_issue_values(row.get("failures")))
    return failures


def issues_from_trace(trace: dict[str, Any]) -> tuple[CandidateIssue, ...]:
    issues: list[CandidateIssue] = []
    validation = trace.get("validation") if isinstance(trace.get("validation"), dict) else {}
    compression = trace.get("compression") if isinstance(trace.get("compression"), dict) else {}
    proxy = trace.get("external_proxy") if isinstance(trace.get("external_proxy"), dict) else {}

    validation_failures = _issue_values(validation.get("failures"))
    if validation_failures:
        issues.append(CandidateIssue.VALIDATION_FAILED)
    if "document_unit_count_changed" in validation_failures:
        issues.append(CandidateIssue.STRUCTURE_CHANGED)
    if "protected_anchor_missing" in validation_failures:
        issues.append(CandidateIssue.ANCHOR_MISSING)

    if not bool(trace.get("compression_accepted")):
        issues.append(CandidateIssue.COMPRESSION_REJECTED)
    compression_status = str(compression.get("status") or "")
    if compression_status == "below_floor":
        issues.append(CandidateIssue.TOO_SHORT)
    elif compression_status == "above_ceiling":
        issues.append(CandidateIssue.TOO_LONG)

    if not bool(trace.get("semantic_safe")):
        issues.append(CandidateIssue.SEMANTIC_DRIFT)
    if trace.get("detector_movement") is False:
        issues.append(CandidateIssue.NO_DETECTOR_MOVEMENT)
    if trace.get("target_gate_passed") is False:
        issues.append(CandidateIssue.NO_TARGET_MOVEMENT)
    text_integrity = trace.get("text_integrity") if isinstance(trace.get("text_integrity"), dict) else {}
    if text_integrity and not bool(text_integrity.get("passed")):
        issues.append(CandidateIssue.TEXT_CORRUPTED)
    ownership_gate = trace.get("ownership_gate") if isinstance(trace.get("ownership_gate"), dict) else {}
    if bool(ownership_gate.get("active")) and not bool(ownership_gate.get("passed")):
        issues.append(CandidateIssue.OWNERSHIP_MISSING)
    outcome = str(trace.get("candidate_outcome") or "")
    if outcome.startswith("generation_failed"):
        issues.append(CandidateIssue.GENERATION_FAILED)

    proxy_reasons = _issue_values(proxy.get("reasons"))
    if proxy_reasons:
        issues.append(CandidateIssue.PROXY_NOT_ACCEPTED)
    if "internal_ai_backfire" in proxy_reasons:
        issues.append(CandidateIssue.INTERNAL_AI_BACKFIRE)
    if "insufficient_topk_drop" in proxy_reasons:
        issues.append(CandidateIssue.INSUFFICIENT_TOPK_DROP)
    topk_failures = _topk_effect_failures(trace)
    if topk_failures:
        issues.append(CandidateIssue.TOPK_CANDIDATE_REJECTED)
    if "no_effect_span_patch" in topk_failures:
        issues.append(CandidateIssue.NO_EFFECT_SPAN_PATCH)
    if "zero_change_candidate" in topk_failures:
        issues.append(CandidateIssue.ZERO_CHANGE_TOPK)
    if "self_report_mismatch" in topk_failures:
        issues.append(CandidateIssue.SELF_REPORT_MISMATCH)
    if "insufficient_span_movement" in topk_failures:
        issues.append(CandidateIssue.INSUFFICIENT_SPAN_MOVEMENT)
    if "writing_quality_collapse" in proxy_reasons:
        issues.append(CandidateIssue.WRITING_QUALITY_COLLAPSE)
    segment_reasons = {
        "segment_ai_fraction_high",
        "segment_ai_or_assisted_fraction_high",
        "segment_human_fraction_low",
        "segment_ai_window_too_large",
        "high_confidence_ai_window_remaining",
    }
    if segment_reasons.intersection(proxy_reasons):
        issues.append(CandidateIssue.SEGMENT_AI_FOOTPRINT)

    unique: list[CandidateIssue] = []
    for issue in issues:
        if issue not in unique:
            unique.append(issue)
    return tuple(unique)


def is_candidate_salvageable(trace: dict[str, Any]) -> bool:
    issues = set(issues_from_trace(trace))
    if CandidateIssue.INTERNAL_AI_BACKFIRE in issues:
        return False
    if CandidateIssue.TEXT_CORRUPTED in issues:
        return False
    if CandidateIssue.GENERATION_FAILED in issues:
        return False
    return True


def select_candidate_index(candidate_evaluations: list[dict[str, Any]]) -> tuple[int, CandidateAction, str]:
    for index, item in enumerate(candidate_evaluations):
        if item.get("strict_selected"):
            return index, CandidateAction.ACCEPT_STRICT, "strict_goal_met"
    for index, item in enumerate(candidate_evaluations):
        if item.get("external_selected"):
            return index, CandidateAction.ACCEPT_EXTERNAL, "external_proxy_accepted"
    return 0, CandidateAction.RETURN_BEST_FOR_REVIEW, "best_generated_candidate_requires_external_review"


def decide_next_action(
    candidate_evaluations: list[dict[str, Any]],
    *,
    has_positive_boundaries: bool,
    tried_actions: set[CandidateAction],
) -> LoopDecision:
    selected_index, action, reason = select_candidate_index(candidate_evaluations)
    if action in {CandidateAction.ACCEPT_STRICT, CandidateAction.ACCEPT_EXTERNAL}:
        return LoopDecision(action=action, source_index=selected_index, issues=(), reason=reason)

    indexed = list(enumerate(candidate_evaluations))
    salvageable = [
        (index, item)
        for index, item in indexed
        if is_candidate_salvageable(item.get("trace") if isinstance(item.get("trace"), dict) else {})
    ]
    latest_trace = candidate_evaluations[-1].get("trace") if isinstance(candidate_evaluations[-1].get("trace"), dict) else {}
    latest_issues = issues_from_trace(latest_trace)
    topk_repair_issues = {
        CandidateIssue.INSUFFICIENT_TOPK_DROP,
        CandidateIssue.TOPK_CANDIDATE_REJECTED,
        CandidateIssue.NO_EFFECT_SPAN_PATCH,
        CandidateIssue.ZERO_CHANGE_TOPK,
        CandidateIssue.SELF_REPORT_MISMATCH,
        CandidateIssue.INSUFFICIENT_SPAN_MOVEMENT,
    }
    if (
        topk_repair_issues.intersection(latest_issues)
        and bool(latest_trace.get("scanner_controlled_executor_available") or latest_trace.get("target_execution_available"))
        and CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR not in tried_actions
    ):
        return LoopDecision(
            action=CandidateAction.SCANNER_CONTROLLED_SPAN_REPAIR,
            source_index=len(candidate_evaluations) - 1,
            issues=latest_issues,
            reason="scanner_controlled_span_repair_after_topk_failure",
        )
    if CandidateIssue.NO_DETECTOR_MOVEMENT in latest_issues or CandidateIssue.NO_TARGET_MOVEMENT in latest_issues:
        if (
            bool(latest_trace.get("target_execution_available"))
            and not bool(latest_trace.get("target_execution_attempted"))
            and CandidateAction.TARGET_EXECUTOR not in tried_actions
        ):
            return LoopDecision(
                action=CandidateAction.TARGET_EXECUTOR,
                source_index=len(candidate_evaluations) - 1,
                issues=latest_issues,
                reason="target_execution_after_no_detector_movement",
            )
        if (
            bool(latest_trace.get("assisted_footprint_executor_available"))
            and CandidateAction.REPAIR_ASSISTED_FOOTPRINT not in tried_actions
        ):
            return LoopDecision(
                action=CandidateAction.REPAIR_ASSISTED_FOOTPRINT,
                source_index=len(candidate_evaluations) - 1,
                issues=latest_issues,
                reason="assisted_footprint_layer_after_target_miss",
            )
        if has_positive_boundaries and CandidateAction.ADAPT_BOUNDARY not in tried_actions:
            return LoopDecision(
                action=CandidateAction.ADAPT_BOUNDARY,
                source_index=len(candidate_evaluations) - 1,
                issues=latest_issues,
                reason="switch_strategy_after_no_detector_movement",
            )
        if has_positive_boundaries and CandidateAction.CONTRAST_BOUNDARY not in tried_actions:
            return LoopDecision(
                action=CandidateAction.CONTRAST_BOUNDARY,
                source_index=len(candidate_evaluations) - 1,
                issues=latest_issues,
                reason="second_strategy_after_no_detector_movement",
            )
        if CandidateAction.PLAIN_REASONING not in tried_actions:
            return LoopDecision(
                action=CandidateAction.PLAIN_REASONING,
                source_index=len(candidate_evaluations) - 1,
                issues=latest_issues,
                reason="plain_reasoning_strategy_after_no_detector_movement",
            )
        return LoopDecision(
            action=CandidateAction.RETURN_BEST_FOR_REVIEW,
            source_index=len(candidate_evaluations) - 1,
            issues=latest_issues,
            reason="stop_after_no_detector_movement",
        )
    for index, item in reversed(salvageable):
        issues = issues_from_trace(item["trace"])
        if CandidateIssue.NO_DETECTOR_MOVEMENT in issues or CandidateIssue.NO_TARGET_MOVEMENT in issues:
            continue
        if CandidateIssue.STRUCTURE_CHANGED in issues and CandidateAction.REPAIR_STRUCTURE not in tried_actions:
            return LoopDecision(
                action=CandidateAction.REPAIR_STRUCTURE,
                source_index=index,
                issues=issues,
                reason="candidate_texture_ok_but_structure_changed",
            )

    hard_contract_issues = {
        CandidateIssue.ANCHOR_MISSING,
        CandidateIssue.COMPRESSION_REJECTED,
        CandidateIssue.TOO_SHORT,
        CandidateIssue.TOO_LONG,
    }
    for index, item in reversed(salvageable):
        issues = issues_from_trace(item["trace"])
        if CandidateIssue.NO_DETECTOR_MOVEMENT in issues or CandidateIssue.NO_TARGET_MOVEMENT in issues:
            continue
        if hard_contract_issues.intersection(issues) and CandidateAction.REPAIR_CONTRACT not in tried_actions:
            return LoopDecision(
                action=CandidateAction.REPAIR_CONTRACT,
                source_index=index,
                issues=issues,
                reason="candidate_failed_generic_contract_invariants",
            )

    for index, item in reversed(salvageable):
        issues = issues_from_trace(item["trace"])
        if CandidateIssue.NO_DETECTOR_MOVEMENT in issues or CandidateIssue.NO_TARGET_MOVEMENT in issues:
            continue
        if CandidateIssue.OWNERSHIP_MISSING in issues and CandidateAction.CLAIM_OWNERSHIP_REPAIR not in tried_actions:
            return LoopDecision(
                action=CandidateAction.CLAIM_OWNERSHIP_REPAIR,
                source_index=index,
                issues=issues,
                reason="claim_ownership_repair_after_human_fraction_failure",
            )
        if CandidateIssue.SEGMENT_AI_FOOTPRINT in issues and CandidateAction.REPAIR_AUTHORSHIP_WINDOWS not in tried_actions:
            if (
                bool(item["trace"].get("assisted_footprint_executor_available"))
                and CandidateAction.REPAIR_ASSISTED_FOOTPRINT not in tried_actions
            ):
                return LoopDecision(
                    action=CandidateAction.REPAIR_ASSISTED_FOOTPRINT,
                    source_index=index,
                    issues=issues,
                    reason="repair_broad_assisted_footprint",
                )
            return LoopDecision(
                action=CandidateAction.REPAIR_AUTHORSHIP_WINDOWS,
                source_index=index,
                issues=issues,
                reason="repair_failed_authorship_windows",
            )

    for index, item in reversed(salvageable):
        issues = issues_from_trace(item["trace"])
        if CandidateIssue.NO_DETECTOR_MOVEMENT in issues or CandidateIssue.NO_TARGET_MOVEMENT in issues:
            continue
        if CandidateIssue.SEMANTIC_DRIFT in issues and CandidateAction.REPAIR_CONTRACT not in tried_actions:
            return LoopDecision(
                action=CandidateAction.REPAIR_CONTRACT,
                source_index=index,
                issues=issues,
                reason="candidate_failed_semantic_contract_invariant",
            )

    if has_positive_boundaries and CandidateAction.ADAPT_BOUNDARY not in tried_actions:
        return LoopDecision(
            action=CandidateAction.ADAPT_BOUNDARY,
            source_index=selected_index,
            issues=issues_from_trace(candidate_evaluations[selected_index]["trace"]),
            reason="use_external_boundaries_for_unresolved_proxy_issues",
        )

    if has_positive_boundaries and CandidateAction.CONTRAST_BOUNDARY not in tried_actions:
        return LoopDecision(
            action=CandidateAction.CONTRAST_BOUNDARY,
            source_index=selected_index,
            issues=issues_from_trace(candidate_evaluations[selected_index]["trace"]),
            reason="contrast_failed_candidate_against_external_boundaries",
        )

    if has_positive_boundaries and CandidateAction.PLAIN_REASONING not in tried_actions:
        return LoopDecision(
            action=CandidateAction.PLAIN_REASONING,
            source_index=selected_index,
            issues=issues_from_trace(candidate_evaluations[selected_index]["trace"]),
            reason="plain_reasoning_broad_prose_candidate_needed",
        )

    if CandidateAction.REPAIR_TARGETED not in tried_actions:
        return LoopDecision(
            action=CandidateAction.REPAIR_TARGETED,
            source_index=selected_index,
            issues=issues_from_trace(candidate_evaluations[selected_index]["trace"]),
            reason="target_unresolved_candidate_issues",
        )

    return LoopDecision(
        action=CandidateAction.RETURN_BEST_FOR_REVIEW,
        source_index=selected_index,
        issues=issues_from_trace(candidate_evaluations[selected_index]["trace"]),
        reason=reason,
    )
