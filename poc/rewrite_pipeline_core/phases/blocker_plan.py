from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BlockerOperationPlanDeps:
    float_env: Callable[[str, float], float]
    blocker_scores: Callable[[dict | None], dict]
    logical_paragraphs: Callable[[str], list[str]]
    join_logical_paragraphs: Callable[[list[str]], str]
    text_word_count: Callable[[str], int]
    paragraph_component_targets: Callable[..., list[dict]]
    paragraph_role: Callable[..., str]
    detect_protected_spans: Callable[[str], list]
    safe_index: Callable[[object, int], int]
    radar_blocker_option_matrix: Callable[..., dict]


def blocker_operation_plan(
    source_text: str,
    raw_json: dict | None,
    *,
    limit: int = 8,
    deps: BlockerOperationPlanDeps,
) -> dict:
    """Compile scanner blockers into hard paragraph operations."""
    blockers = deps.blocker_scores(raw_json)
    paragraphs = deps.logical_paragraphs(source_text)
    if not paragraphs:
        return {"enabled": False, "reason": "no_paragraphs", "blockers": blockers, "operations": []}

    active = {
        key: value for key, value in blockers.items()
        if isinstance(value, (int, float)) and float(value) >= 60.0
    }
    if not active:
        return {"enabled": False, "reason": "no_active_blockers", "blockers": blockers, "operations": []}

    protected = deps.detect_protected_spans(source_text)

    def protected_in_paragraph(index: int) -> bool:
        before = deps.join_logical_paragraphs(paragraphs[:index])
        start = len(before) + (2 if before else 0)
        end = start + len(paragraphs[index])
        return any(span.start_char >= start and span.end_char <= end for span in protected)

    targets = deps.paragraph_component_targets(source_text, raw_json or {}, limit=max(limit * 2, 6))
    operations = []
    for target in targets:
        index = int(target.get("index", 0) or 0)
        if index < 0 or index >= len(paragraphs):
            continue
        paragraph = paragraphs[index]
        words = deps.text_word_count(paragraph)
        if words < 20:
            continue
        drivers = target.get("drivers") or {}
        role = target.get("role") or deps.paragraph_role(paragraph, drivers, is_last=index == len(paragraphs) - 1)
        has_protected = protected_in_paragraph(index)
        generic_hits = int(drivers.get("generic_assertion_hits") or 0)
        concrete_hits = int(drivers.get("concrete_anchor_hits") or 0)
        source_gap = bool(drivers.get("source_gap"))
        generic_density = generic_hits / max(words / 100.0, 1.0)

        reasons = []
        if blockers.get("unsupported_claim_risk", 0.0) >= 75 and source_gap:
            reasons.append("unsupported_claim_risk")
        if blockers.get("broad_claim_risk", 0.0) >= 75 and generic_hits >= 3:
            reasons.append("broad_claim_risk")
        if blockers.get("generic_assertion_risk", 0.0) >= 75 and generic_density >= 3.0:
            reasons.append("generic_assertion_risk")
        if blockers.get("topk_pattern", 0.0) >= 75 and target.get("target_sentences"):
            reasons.append("topk_pattern")
        if blockers.get("lived_detail_risk", 0.0) >= 70 and concrete_hits <= 2:
            reasons.append("lived_detail_risk")
        if not reasons:
            continue

        if role in {"human_anchor_rich", "technical_process_rich"}:
            operation = "preserve_micro_texture"
        elif has_protected:
            operation = "narrow_protected_paragraph"
        elif role == "conclusion_template_risk":
            operation = "compress_or_delete"
        elif "generic_assertion_risk" in reasons or "unsupported_claim_risk" in reasons:
            operation = "delete_or_compress"
        elif "broad_claim_risk" in reasons:
            operation = "claim_narrow"
        else:
            operation = "topk_texture_patch"

        priority = (
            float(target.get("score") or 0.0)
            + len(reasons) * 10.0
            + generic_density * 2.0
            + (8.0 if source_gap else 0.0)
            - concrete_hits * 0.5
            - (8.0 if has_protected else 0.0)
        )
        operations.append({
            "paragraph_index": index,
            "operation": operation,
            "role": role,
            "priority": round(priority, 3),
            "blockers": reasons,
            "has_protected_anchor": has_protected,
            "word_count": words,
            "generic_density": round(generic_density, 3),
            "drivers": drivers,
            "preview": paragraph[:360],
        })

    operations.sort(key=lambda item: item["priority"], reverse=True)
    selected_operations = operations[:max(1, int(limit or 1))]
    decisions = block_level_decisions(
        selected_operations,
        source_text,
        blockers=blockers,
        deps=deps,
    )
    option_matrix = deps.radar_blocker_option_matrix(raw_json, limit=limit)
    return {
        "enabled": True,
        "kind": "blocker_operation_plan",
        "blockers": blockers,
        "active_blockers": active,
        "operations": selected_operations,
        "block_decisions": decisions,
        "radar_option_matrix": option_matrix,
        "policy": [
            "reinforce source-worthy claims before removal when public evidence can help",
            "delete or compress score-dragging generic paragraphs before texture repair",
            "narrow unsupported/broad claims before adding sources or author language",
            "preserve protected anchors; never delete a protected paragraph automatically",
            "rank candidate selection by blocker elimination before cosmetic score movement",
        ],
    }


def block_level_decisions(
    operations: list[dict],
    source_text: str,
    *,
    blockers: dict | None = None,
    deps: BlockerOperationPlanDeps,
) -> list[dict]:
    """Classify each risky block before candidate generation."""
    blockers = blockers or {}
    max_search_calls = max(
        0,
        min(5, int(deps.float_env("DRAFTPROOF_SOURCE_SEARCH_MAX_CALLS_PER_RUN", 5.0))),
    )
    search_reserved = 0
    decisions: list[dict] = []
    for op in operations or []:
        paragraph_index = deps.safe_index(op.get("paragraph_index"), -1)
        role = str(op.get("role") or "")
        drivers = op.get("drivers") if isinstance(op.get("drivers"), dict) else {}
        blocker_keys = set(op.get("blockers") or [])
        word_count = int(op.get("word_count") or 0)
        generic_density = float(op.get("generic_density") or 0.0)
        generic_hits = int(drivers.get("generic_assertion_hits") or 0)
        concrete_hits = int(drivers.get("concrete_anchor_hits") or 0)
        source_gap = bool(drivers.get("source_gap"))
        has_protected = bool(op.get("has_protected_anchor"))
        has_reinforce_value = bool(
            word_count >= 25
            and (
                concrete_hits >= 1
                or role in {"source_summary_heavy", "generic_claim_heavy"}
                or "source_grounding_risk" in blocker_keys
                or "unsupported_claim_risk" in blocker_keys
            )
        )
        heavy_generic_drag = bool(
            generic_density >= 3.0
            or generic_hits >= 4
            or role == "conclusion_template_risk"
        )

        decision = "preserve"
        reason = "no_high_value_operation"
        uses_search = False
        allowed_operations = ["preserve"]

        if has_protected or role in {"human_anchor_rich", "technical_process_rich"}:
            decision = "preserve_micro_repair"
            reason = "protected_or_high_value_human_block"
            allowed_operations = ["micro_texture_patch", "claim_narrow"]
        elif source_gap and has_reinforce_value and search_reserved < max_search_calls:
            decision = "reinforce_with_public_source"
            reason = "source_gap_with_salvageable_claim"
            uses_search = True
            search_reserved += 1
            allowed_operations = ["source_reinforce", "claim_narrow"]
        elif source_gap and heavy_generic_drag and concrete_hits <= 1:
            decision = "remove_or_compress"
            reason = "unsupported_generic_score_drag"
            allowed_operations = ["delete_paragraph", "compress_paragraph"]
        elif "broad_claim_risk" in blocker_keys or "generic_assertion_risk" in blocker_keys:
            decision = "claim_narrow"
            reason = "broad_or_generic_claim_can_be_limited"
            allowed_operations = ["claim_narrow", "compress_paragraph"]
        elif "topk_pattern" in blocker_keys:
            decision = "topk_texture_patch"
            reason = "predictable_local_texture"
            allowed_operations = ["micro_texture_patch"]
        elif source_gap:
            decision = "ask_author_context"
            reason = "source_gap_not_safe_to_reinforce_or_remove"
            allowed_operations = ["ask_author"]

        decisions.append({
            "paragraph_index": paragraph_index,
            "decision": decision,
            "reason": reason,
            "role": role,
            "blockers": sorted(blocker_keys),
            "uses_source_search": uses_search,
            "source_search_slot": search_reserved if uses_search else None,
            "allowed_operations": allowed_operations,
            "salvageable": decision in {
                "reinforce_with_public_source",
                "claim_narrow",
                "topk_texture_patch",
                "preserve_micro_repair",
            },
            "fallback_if_failed": (
                "remove_or_compress"
                if decision == "reinforce_with_public_source" and not has_protected
                else "ask_author_context"
                if decision in {"claim_narrow", "topk_texture_patch"}
                else "preserve"
            ),
            "budget": {
                "source_search_reserved": search_reserved,
                "source_search_max": max_search_calls,
            },
        })
    return decisions
