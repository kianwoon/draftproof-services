from __future__ import annotations

from typing import Any

from .finding_pattern import classify_finding_pattern
from .plan import Plan
from .prose_repair_rules import risky_source_shape_guidance, risky_source_terms


def build_paragraph_planner_brief(plan: Plan) -> dict[str, Any]:
    strategy = plan.paragraph_strategy if isinstance(plan.paragraph_strategy, dict) else {}
    grounding = strategy.get("author_proxy_grounding") if isinstance(strategy.get("author_proxy_grounding"), dict) else {}
    pattern = classify_finding_pattern(plan)
    return _drop_empty({
        "repair_unit": strategy.get("repair_unit"),
        "target_sentence_range": strategy.get("target_sentence_range"),
        "paragraph_diagnosis": _paragraph_diagnosis(plan),
        "finding_pattern": pattern,
        "paragraph_route": pattern.get("pattern_id") or "source_order_paragraph_rebuild",
        "human_route": _pattern_route(pattern),
        "flow_plan": _source_order_flow(plan),
        "semantic_role_map": _source_order_roles(plan),
        "finding_summary": _finding_summary(plan),
        "author_proxy_boundary": _drop_empty({
            "required": grounding.get("required"),
            "reason": grounding.get("reason"),
            "writer_instruction": grounding.get("writer_instruction"),
            "boundary": "Use proxy or neighbor material only to resolve a local anchor gap; do not import new claims.",
        }),
    })


def _paragraph_diagnosis(plan: Plan) -> str:
    tags = _tag_counts(plan)
    issues = []
    if tags.get("packed_list"):
        issues.append("packed source material")
    if tags.get("predictable_start") or tags.get("paragraph_rhythm"):
        issues.append("repeated paragraph rhythm")
    if tags.get("context_anchor_gap") or tags.get("broad_claim") or tags.get("unsupported_claim_gap"):
        issues.append("weak source-to-claim anchoring")
    return (
        "Sentence findings are paragraph-flow symptoms: "
        + (", ".join(issues) if issues else "local wording needs source-order repair")
        + ". Rewrite the paragraph as one source-grounded movement."
    )


def _source_order_route() -> dict[str, Any]:
    return {
        "route_type": "SOURCE_ORDER_PARAGRAPH_REBUILD",
        "movement": "opening source beat -> grouped development beats -> closing source beat",
        "paragraph_jobs": [
            "Carry the opening source beat without adding a new topic.",
            "Group middle source beats by meaning, not by scanner finding.",
            "Carry the closing source beat as the paragraph's own limit or contrast.",
        ],
        "avoid": ["sentence-by-sentence repair", "new claims", "neighbor paragraph topics", "policy recommendation"],
    }


def _pattern_route(pattern: dict[str, Any]) -> dict[str, Any]:
    return {
        "route_type": str(pattern.get("pattern_id") or "source_order_paragraph_rebuild").upper(),
        "movement": pattern.get("movement") or "opening source beat -> grouped development beats -> closing source beat",
        "paragraph_jobs": list(pattern.get("paragraph_jobs") or []),
        "avoid": list(pattern.get("avoid") or []),
    }


def _source_order_flow(plan: Plan) -> list[dict[str, Any]]:
    actions = [action for action in plan.actions if action.source_text.strip()]
    if not actions:
        return []
    middle = actions[1:-1] if len(actions) > 2 else actions
    return [
        _flow_row("opening", actions[:1], "carry the opening source beat"),
        _flow_row("development", middle, "group the middle source beats into paragraph flow"),
        _flow_row("closing", actions[-1:], "carry the closing source beat without adding a new conclusion"),
    ]


def _flow_row(step_id: str, actions: list[Any], function: str) -> dict[str, Any]:
    terms = []
    source_text = " ".join(str(action.source_text or "") for action in actions)
    for action in actions:
        terms.extend(action.preserve_terms[:4])
    return _drop_empty({
        "step_id": step_id,
        "function": function,
        "source_basis": [action.sentence_id for action in actions],
        "must_include": _safe_must_include_terms(terms, source_text)[:10],
        "risky_source_shapes": _risky_source_shapes(source_text),
    })


def _source_order_roles(plan: Plan) -> dict[str, Any]:
    terms = []
    for action in plan.actions:
        terms.extend(action.preserve_terms[:5])
    return {
        "source_order_rule": "Use only selected paragraph source beats in their original order.",
        "paragraph_terms": _safe_must_include_terms(terms, " ".join(action.source_text for action in plan.actions))[:18],
    }


def _safe_must_include_terms(terms: list[str], source_text: str) -> list[str]:
    risky = _risky_exact_terms(source_text)
    return [term for term in _dedupe(terms) if term.casefold() not in risky]


def _risky_exact_terms(source_text: str) -> set[str]:
    return risky_source_terms(source_text)


def _risky_source_shapes(source_text: str) -> list[str]:
    return risky_source_shape_guidance(source_text)


def _finding_summary(plan: Plan) -> dict[str, Any]:
    return {
        "diagnostic_use_only": "Sentence findings are inputs to the planner; writer must not fix sentence-by-sentence.",
        "tag_counts": _tag_counts(plan),
    }


def _tag_counts(plan: Plan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in plan.actions:
        for tag in action.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows


def _drop_empty(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}
