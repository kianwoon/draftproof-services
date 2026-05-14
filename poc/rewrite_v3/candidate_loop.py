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
    WRITING_QUALITY_COLLAPSE = "writing_quality_collapse"
    PROXY_NOT_ACCEPTED = "proxy_not_accepted"


class CandidateAction(str, Enum):
    ACCEPT_STRICT = "accept_strict"
    ACCEPT_EXTERNAL = "accept_external"
    REPAIR_CONTRACT = "repair_contract"
    REPAIR_STRUCTURE = "repair_structure"
    CONTRAST_BOUNDARY = "contrast_boundary"
    PLAIN_REASONING = "plain_reasoning"
    REPAIR_TARGETED = "repair_targeted"
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

    proxy_reasons = _issue_values(proxy.get("reasons"))
    if proxy_reasons:
        issues.append(CandidateIssue.PROXY_NOT_ACCEPTED)
    if "internal_ai_backfire" in proxy_reasons:
        issues.append(CandidateIssue.INTERNAL_AI_BACKFIRE)
    if "insufficient_topk_drop" in proxy_reasons:
        issues.append(CandidateIssue.INSUFFICIENT_TOPK_DROP)
    if "writing_quality_collapse" in proxy_reasons:
        issues.append(CandidateIssue.WRITING_QUALITY_COLLAPSE)

    unique: list[CandidateIssue] = []
    for issue in issues:
        if issue not in unique:
            unique.append(issue)
    return tuple(unique)


def is_candidate_salvageable(trace: dict[str, Any]) -> bool:
    issues = set(issues_from_trace(trace))
    if CandidateIssue.INTERNAL_AI_BACKFIRE in issues:
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
    for index, item in reversed(salvageable):
        issues = issues_from_trace(item["trace"])
        if CandidateIssue.STRUCTURE_CHANGED in issues and CandidateAction.REPAIR_STRUCTURE not in tried_actions:
            return LoopDecision(
                action=CandidateAction.REPAIR_STRUCTURE,
                source_index=index,
                issues=issues,
                reason="candidate_texture_ok_but_structure_changed",
            )

    contract_issues = {
        CandidateIssue.ANCHOR_MISSING,
        CandidateIssue.COMPRESSION_REJECTED,
        CandidateIssue.TOO_SHORT,
        CandidateIssue.TOO_LONG,
        CandidateIssue.SEMANTIC_DRIFT,
    }
    for index, item in reversed(salvageable):
        issues = issues_from_trace(item["trace"])
        if contract_issues.intersection(issues) and CandidateAction.REPAIR_CONTRACT not in tried_actions:
            return LoopDecision(
                action=CandidateAction.REPAIR_CONTRACT,
                source_index=index,
                issues=issues,
                reason="candidate_failed_generic_contract_invariants",
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
