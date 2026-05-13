"""Candidate failure diagnostics for rewrite V2.

The selector decides whether a candidate can be applied. This module explains
why candidates were rejected so strategy tuning can be based on failure classes
instead of one-off log inspection.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


FIXABLE_CONTRACT_DRIFT = "fixable_contract_drift"
HARD_ANCHOR_LOSS = "hard_anchor_loss"
SEMANTIC_LOSS = "semantic_loss"
DETECTOR_NOT_SAFE = "detector_not_safe"
GOAL_NOT_MET = "goal_not_met"
GENERATION_FAILED = "generation_failed"
LOCAL_QUALITY_REJECTED = "local_quality_rejected"
UNKNOWN_REJECTED = "unknown_rejected"
NOT_FAILED = "not_failed"


def _decision(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("decision")
    return value if isinstance(value, dict) else {}


def _goal(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("goal")
    return value if isinstance(value, dict) else {}


def _local_filter_failures(row: dict[str, Any]) -> list[str]:
    return [
        str(reason)
        for reason in (row.get("local_filter_failures") or [])
        if str(reason or "").strip()
    ]


def _semantic_reasons(row: dict[str, Any]) -> list[str]:
    return [
        str(reason)
        for reason in (row.get("semantic_reasons") or [])
        if str(reason or "").strip()
    ]


def _has_fixable_contract_drift(failures: list[str]) -> bool:
    fixable_markers = (
        "citation_lost:",
        "protected_span_lost:",
        "heading_lost:",
        "heading_not_preserved",
        "protected_quote_lost:",
        "required_term_lost:",
    )
    return bool(failures) and any(marker in reason for reason in failures for marker in fixable_markers)


def _has_hard_anchor_loss(failures: list[str], row: dict[str, Any]) -> bool:
    if row.get("protected_anchors_safe") is False:
        return True
    hard_markers = (
        "direct_quote_lost",
        "numeric_anchor_lost",
        "hard_anchor_lost",
        "quote_lost",
    )
    return any(marker in reason for reason in failures for marker in hard_markers)


def _external_proxy_unsafe(row: dict[str, Any]) -> bool:
    proxy = _goal(row).get("external_detector_proxy")
    return isinstance(proxy, dict) and proxy.get("safe") is False


def diagnose_candidate_failure(row: dict[str, Any]) -> dict[str, Any]:
    """Return a stable failure class for a generated candidate row."""
    decision = _decision(row)
    lane = str(decision.get("lane") or "")
    reason = str(decision.get("reason") or "")
    failures = _local_filter_failures(row)
    semantic_reasons = _semantic_reasons(row)

    if lane and lane != "REJECT" and reason == "strict_goal_met":
        return {"failure_class": NOT_FAILED, "failure_reasons": []}
    if not lane and not failures and not reason:
        return {"failure_class": GENERATION_FAILED, "failure_reasons": ["candidate_not_evaluated"]}

    if _has_hard_anchor_loss(failures, row):
        return {"failure_class": HARD_ANCHOR_LOSS, "failure_reasons": failures[:8]}
    if _has_fixable_contract_drift(failures):
        return {"failure_class": FIXABLE_CONTRACT_DRIFT, "failure_reasons": failures[:8]}
    if "surface_quality:" in " ".join(failures) or "banned_phrase:" in " ".join(failures):
        return {"failure_class": LOCAL_QUALITY_REJECTED, "failure_reasons": failures[:8]}
    if row.get("semantic_safe") is False or "semantic" in reason:
        return {
            "failure_class": SEMANTIC_LOSS,
            "failure_reasons": (semantic_reasons or [reason])[:8],
        }
    if reason == "external_detector_proxy_not_safe" or _external_proxy_unsafe(row):
        return {"failure_class": DETECTOR_NOT_SAFE, "failure_reasons": [reason]}
    if reason in {"partial_progress_not_success", "candidate_failed_strict_detector_safe_goal"}:
        return {"failure_class": GOAL_NOT_MET, "failure_reasons": [reason]}
    if lane == "REJECT":
        return {"failure_class": UNKNOWN_REJECTED, "failure_reasons": (failures or [reason])[:8]}
    return {"failure_class": NOT_FAILED, "failure_reasons": []}


def annotate_candidate_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    diagnostic = diagnose_candidate_failure(row)
    return {**row, **diagnostic}


def summarize_candidate_diagnostics(rows: list[dict[str, Any]], generated_count: int = 0) -> dict[str, Any]:
    annotated = [annotate_candidate_diagnostics(row) for row in rows]
    counts = Counter(str(row.get("failure_class") or UNKNOWN_REJECTED) for row in annotated)
    if generated_count == 0 and not rows:
        counts[GENERATION_FAILED] += 1
    return {
        "failure_class_counts": dict(sorted(counts.items())),
        "primary_failure_class": counts.most_common(1)[0][0] if counts else None,
        "fixable_contract_drift_count": counts.get(FIXABLE_CONTRACT_DRIFT, 0),
        "detector_not_safe_count": counts.get(DETECTOR_NOT_SAFE, 0),
        "semantic_loss_count": counts.get(SEMANTIC_LOSS, 0),
        "hard_anchor_loss_count": counts.get(HARD_ANCHOR_LOSS, 0),
    }
