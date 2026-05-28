from __future__ import annotations

from typing import Any


def retryable_no_change_result(result: Any) -> bool:
    selected = getattr(result, "selected", None)
    diagnostics = list(getattr(result, "candidate_diagnostics", []) or [])
    generated_rows = [row for row in diagnostics if row.get("source") != "source_preserved"]
    if selected is None:
        return bool(generated_rows) and any(_selector_did_not_make_valid_choice(row) for row in generated_rows)
    if getattr(selected, "source", None) != "source_preserved":
        return False
    if _author_proxy_grounding_required(result):
        return True
    if not generated_rows:
        return True
    if any(_selector_did_not_make_valid_choice(row) for row in generated_rows):
        return True
    return any(_blocked_row_moved_scanner(row) for row in generated_rows if row.get("blockers"))


def no_change_retry_status(result: Any, *, will_retry: bool) -> str:
    if will_retry:
        return "no_change_retryable"
    if retryable_no_change_result(result):
        return "no_change_retry_exhausted"
    return "no_change"


def no_change_retry_message(paragraph_id: str, *, will_retry: bool) -> str:
    if will_retry:
        return f"V6 paragraph {paragraph_id} made no accepted change; retrying later"
    return f"V6 paragraph {paragraph_id} made no change"


def _blocked_row_moved_scanner(row: dict[str, Any]) -> bool:
    try:
        finding_drop = int(row.get("finding_drop") or 0)
        risk_drop = float(row.get("risk_drop") or 0.0)
    except (TypeError, ValueError):
        return False
    return finding_drop > 0 or risk_drop > 0


def _selector_did_not_make_valid_choice(row: dict[str, Any]) -> bool:
    source = str(row.get("selector_source") or "")
    return source.startswith("invalid_selector_source_preserved") or source.startswith("selector_required_missing")


def _author_proxy_grounding_required(result: Any) -> bool:
    plan = getattr(result, "plan", None)
    strategy = getattr(plan, "paragraph_strategy", {}) if plan is not None else {}
    if not isinstance(strategy, dict):
        return False
    grounding = strategy.get("author_proxy_grounding")
    return isinstance(grounding, dict) and bool(grounding.get("required"))
