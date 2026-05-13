"""Layer attempt ledger for rewrite V2 candidate generation."""

from __future__ import annotations

from typing import Any


SKIPPED_INAPPLICABLE_REASONS = {
    "blocked_by_content_router",
    "disabled_by_config",
    "no_targets",
    "not_applicable",
    "strategy_family_blocked",
}


def record_layer_attempt(
    attempts: list[dict[str, Any]] | None,
    *,
    layer: str,
    status: str,
    reason: str,
    allowed: bool | None = None,
    applicable: bool | None = None,
    generated_count: int = 0,
    candidate_count_before: int | None = None,
    candidate_count_after: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append a normalized attempt row when a caller provided a ledger."""
    if attempts is None:
        return
    row: dict[str, Any] = {
        "layer": str(layer or "unknown"),
        "status": str(status or "unknown"),
        "reason": str(reason or "unspecified"),
        "generated_count": max(0, int(generated_count or 0)),
    }
    if allowed is not None:
        row["allowed"] = bool(allowed)
    if applicable is not None:
        row["applicable"] = bool(applicable)
    if candidate_count_before is not None:
        row["candidate_count_before"] = max(0, int(candidate_count_before or 0))
    if candidate_count_after is not None:
        row["candidate_count_after"] = max(0, int(candidate_count_after or 0))
    if metadata:
        row["metadata"] = dict(metadata)
    attempts.append(row)


def summarize_layer_attempts(attempts: list[dict[str, Any]] | None) -> dict[str, Any]:
    rows = list(attempts or [])
    by_layer: dict[str, dict[str, Any]] = {}
    for row in rows:
        layer = str(row.get("layer") or "unknown")
        status = str(row.get("status") or "unknown")
        summary = by_layer.setdefault(layer, {
            "attempts": 0,
            "generated_count": 0,
            "statuses": {},
            "reasons": {},
            "allowed": False,
            "applicable": False,
        })
        summary["attempts"] += 1
        summary["generated_count"] += max(0, int(row.get("generated_count") or 0))
        summary["statuses"][status] = summary["statuses"].get(status, 0) + 1
        reason = str(row.get("reason") or "unspecified")
        summary["reasons"][reason] = summary["reasons"].get(reason, 0) + 1
        if row.get("allowed") is True:
            summary["allowed"] = True
        if row.get("applicable") is True:
            summary["applicable"] = True
    return {
        "attempt_count": len(rows),
        "layers": {
            layer: {
                **summary,
                "statuses": dict(sorted(summary["statuses"].items())),
                "reasons": dict(sorted(summary["reasons"].items())),
            }
            for layer, summary in sorted(by_layer.items())
        },
    }


def layer_marked_inapplicable(attempts: list[dict[str, Any]] | None, layer: str) -> tuple[bool, str | None]:
    layer_rows = [row for row in attempts or [] if str(row.get("layer") or "") == layer]
    if not layer_rows:
        return False, None
    if any(row.get("generated_count") for row in layer_rows):
        return False, None
    if any(row.get("applicable") is True for row in layer_rows):
        return False, None
    for row in layer_rows:
        reason = str(row.get("reason") or "")
        if row.get("applicable") is False or reason in SKIPPED_INAPPLICABLE_REASONS:
            return True, reason or "not_applicable"
    return False, None
