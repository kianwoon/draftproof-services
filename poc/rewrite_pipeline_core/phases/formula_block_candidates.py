from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FormulaBlockCandidateDeps:
    env_flag: Callable[[str, bool], bool]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]


def formula_block_map_removal_candidates(
    source_text: str,
    block_map: dict | None,
    *,
    limit: int = 3,
    deps: FormulaBlockCandidateDeps,
) -> list[tuple[str, str, dict]]:
    """Remove low-value high-drag blocks identified by the formula block map."""
    if not deps.env_flag("DRAFTPROOF_FORMULA_BLOCK_MAP_REMOVAL", True):
        return []
    paragraphs = deps.logical_paragraphs(source_text)
    if len(paragraphs) < 3:
        return []
    removable = [
        row for row in (block_map or {}).get("blocks") or []
        if isinstance(row, dict)
        and row.get("recommended_portfolio_action") == "remove_candidate"
        and row.get("remove_value_loss_risk") == "low"
        and not row.get("protected")
        and not row.get("unique_core_claim")
        and isinstance(row.get("block_index"), int)
        and 0 <= int(row.get("block_index")) < len(paragraphs)
    ]
    removable.sort(
        key=lambda row: (
            float(row.get("weighted_drag") or 0.0),
            float(row.get("generic_density") or 0.0),
            float(row.get("human_anchor_deficit") or 0.0),
        ),
        reverse=True,
    )
    if not removable:
        return []
    limit = max(1, int(limit or 1))
    source_norm = str(source_text or "").strip()
    candidates: list[tuple[str, str, dict]] = []
    seen = {source_norm}

    def add(strategy: str, remove_indexes: list[int]) -> None:
        indexes = sorted(set(remove_indexes))
        if not indexes or len(indexes) >= len(paragraphs):
            return
        next_paragraphs = [
            paragraph for idx, paragraph in enumerate(paragraphs)
            if idx not in indexes
        ]
        candidate = deps.join_logical_paragraphs(next_paragraphs)
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        removed_rows = [
            row for row in removable
            if int(row.get("block_index")) in indexes
        ]
        candidates.append((
            strategy,
            candidate,
            {
                "operation": "formula_low_value_block_remove",
                "portfolio_operation": "low_value_remove",
                "removed_paragraph_indexes": indexes,
                "removed_blocks": removed_rows,
                "targeted_drivers": [
                    "patchwork_expansion",
                    "semantic_uniformity",
                    "ai_likelihood",
                    "rewrite_smoothness",
                ],
                "formula_block_map_removal": True,
            },
        ))

    for row in removable[:limit]:
        add(f"formula_low_value_block_remove_p{int(row['block_index']) + 1}", [int(row["block_index"])])
        if len(candidates) >= limit:
            return candidates[:limit]
    if len(removable) >= 2 and len(candidates) < limit:
        add(
            "formula_low_value_block_remove_top2",
            [int(row["block_index"]) for row in removable[:2]],
        )
    return candidates[:limit]
