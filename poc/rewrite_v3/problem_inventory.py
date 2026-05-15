"""Problem-inventory helpers for rewrite V3.

The scanner owns problem discovery. V3 only converts the structured inventory
into executable strategy steps and target filters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .scanner_contract import ScanContract


@dataclass(frozen=True)
class ProblemStrategyStep:
    strategy_id: str
    target_issue: str
    editable_scope: str
    problem_group_ids: tuple[str, ...] = field(default_factory=tuple)
    target_ids: tuple[str, ...] = field(default_factory=tuple)
    max_candidates: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def problem_groups(contract: ScanContract) -> list[dict[str, Any]]:
    return [row for row in contract.problem_groups if isinstance(row, dict)]


def target_ids_for_strategy(contract: ScanContract, strategy_id: str) -> set[str]:
    target_ids: set[str] = set()
    for group in problem_groups(contract):
        allowed = {str(item) for item in group.get("allowed_operations") or []}
        if strategy_id not in allowed:
            continue
        target_ids.update(str(item) for item in group.get("target_ids") or [] if str(item))
    return target_ids


def build_problem_strategy_steps(contract: ScanContract) -> list[ProblemStrategyStep]:
    groups = problem_groups(contract)
    if not groups:
        return []

    ordered_strategies = (
        "unit_preserving_prune_bridge",
        "paragraph_preserving_broad_reconstruction",
        "citation_preserving_window_repair",
        "paragraph_surgery",
        "protected_section_rewrite",
        "chunk_reconstruction",
        "authorship_window_repair",
        "avoid_aggressive_rewrite",
    )
    priority = {name: index for index, name in enumerate(ordered_strategies)}
    rows: dict[str, dict[str, Any]] = {}
    for group in groups:
        group_id = str(group.get("group_id") or "")
        shape = str(group.get("problem_shape") or "targeted_problem")
        scope = str(group.get("scope_level") or "target_profile")
        blocked = {str(item) for item in group.get("blocked_operations") or []}
        for operation in group.get("allowed_operations") or []:
            strategy_id = str(operation)
            if not strategy_id or strategy_id in blocked:
                continue
            current = rows.setdefault(strategy_id, {
                "strategy_id": strategy_id,
                "target_issue": shape,
                "editable_scope": scope,
                "problem_group_ids": [],
                "target_ids": [],
                "score": 0.0,
            })
            current["problem_group_ids"].append(group_id)
            current["target_ids"].extend(str(item) for item in group.get("target_ids") or [] if str(item))
            current["score"] += (
                1.0
                + _number(group.get("anchor_pressure")) * 0.2
                + _number(group.get("semantic_edit_cost")) * 0.1
                - priority.get(strategy_id, 50) * 0.03
            )

    sorted_rows = sorted(
        rows.values(),
        key=lambda row: (priority.get(str(row["strategy_id"]), 50), -_number(row.get("score"))),
    )
    steps: list[ProblemStrategyStep] = []
    for row in sorted_rows:
        strategy_id = str(row["strategy_id"])
        if strategy_id == "avoid_aggressive_rewrite":
            steps.append(ProblemStrategyStep(
                strategy_id="portfolio_selection",
                target_issue="low_footprint_no_aggressive_rewrite",
                editable_scope="candidate_set",
                problem_group_ids=tuple(row["problem_group_ids"]),
                target_ids=tuple(dict.fromkeys(row["target_ids"])),
            ))
            continue
        steps.append(ProblemStrategyStep(
            strategy_id=strategy_id,
            target_issue=str(row.get("target_issue") or "scanner_problem"),
            editable_scope=str(row.get("editable_scope") or "target_profile"),
            problem_group_ids=tuple(row["problem_group_ids"]),
            target_ids=tuple(dict.fromkeys(row["target_ids"])),
        ))
    return steps


def unresolved_problem_groups(
    *,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    before_groups = (before or {}).get("problem_groups") if isinstance(before, dict) else []
    after_groups = (after or {}).get("problem_groups") if isinstance(after, dict) else []
    before_ids = {
        str(group.get("group_id") or "")
        for group in before_groups or []
        if isinstance(group, dict)
    }
    unresolved = []
    for group in after_groups or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id") or "")
        if not group_id or group_id in before_ids or str(group.get("problem_shape") or "") != "low_footprint":
            unresolved.append(group)
    return unresolved


__all__ = [
    "ProblemStrategyStep",
    "build_problem_strategy_steps",
    "problem_groups",
    "target_ids_for_strategy",
    "unresolved_problem_groups",
]
