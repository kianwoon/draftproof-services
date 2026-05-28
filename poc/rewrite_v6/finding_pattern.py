from __future__ import annotations

import re
from typing import Any

from .plan import Plan


def classify_finding_pattern(plan: Plan) -> dict[str, Any]:
    text = _pattern_text(plan)
    tags = _tag_counts(plan)
    pattern_id = _pattern_id(text, tags)
    return {
        "pattern_id": pattern_id,
        "confidence": _confidence(pattern_id, tags),
        "tag_counts": tags,
        **_route(pattern_id),
    }


def _pattern_id(text: str, tags: dict[str, int]) -> str:
    lowered = text.casefold()
    if _has_benefit_risk_shape(lowered):
        return "benefit_risk_contrast"
    if _has_old_new_mismatch_shape(lowered):
        return "old_model_current_mismatch"
    if _has_turning_point_shape(lowered):
        return "turning_point_conclusion"
    if _has_recommendation_process_shape(lowered):
        return "recommendation_process"
    if tags.get("author_anchor_gap") or tags.get("unsupported_claim_gap"):
        return "claim_grounding_gap"
    if tags.get("context_anchor_gap") and tags.get("packed_list"):
        return "anchor_then_compact_development"
    if tags.get("packed_list", 0) >= 2:
        return "packed_list_flow_rebuild"
    return "claim_reason_consequence"


def _has_benefit_risk_shape(text: str) -> bool:
    positive = r"\b(?:opportunit(?:y|ies)|benefits?|support|help|useful|used\s+well)\b"
    negative = r"\b(?:risks?|danger|dependent|overly|without\s+understanding|difficult|fairness|integrity|concerns?)\b"
    return bool(re.search(positive, text) and re.search(negative, text))


def _has_old_new_mismatch_shape(text: str) -> bool:
    old_side = r"\b(?:in\s+the\s+past|old|traditional|used\s+to|still\s+exists)\b"
    mismatch = r"\b(?:no\s+longer|not\s+fully|does\s+not\s+reflect|mismatch|changing\s+faster)\b"
    habits_mismatch = r"\b(?:old\s+habits?|current\s+\w+\s+system)\b[\s\S]{0,600}\b(?:modern\s+world|today|current|now)\b"
    return bool(
        re.search(old_side, text) and re.search(mismatch, text)
        or re.search(habits_mismatch, text)
    )


def _has_recommendation_process_shape(text: str) -> bool:
    return bool(re.search(r"\b(?:should|must|need(?:s)?\s+to|include|process|feedback|improvement)\b", text))


def _has_turning_point_shape(text: str) -> bool:
    return bool(re.search(r"\b(?:turning\s+point|either|goal\s+should|reject|protect|redesign)\b", text))


def _confidence(pattern_id: str, tags: dict[str, int]) -> str:
    if pattern_id in {"benefit_risk_contrast", "old_model_current_mismatch"}:
        return "high"
    if tags.get("author_anchor_gap") or tags.get("unsupported_claim_gap"):
        return "medium"
    return "low"


def _route(pattern_id: str) -> dict[str, Any]:
    routes = {
        "benefit_risk_contrast": {
            "movement": "useful support -> good-use condition -> possible misuse risk -> consequence",
            "paragraph_jobs": [
                "State the useful support without listing every benefit as a separate beat.",
                "Keep the good-use condition close to the benefit.",
                "Introduce the possible risk with a direct risk sentence, such as the risk is that the actor may become dependent, not an if/may double hedge.",
                "Close on the practical consequence already in the source.",
            ],
            "avoid": [
                "balanced essay template",
                "turning possible risk into certain harm",
                "awkward double hedge such as a danger arises if students may",
                "one long benefits sentence followed by one long risks sentence",
                "generic concern tail",
            ],
        },
        "old_model_current_mismatch": {
            "movement": "pace of change -> old model -> old proof of learning -> mismatch today",
            "paragraph_jobs": [
                "Name the change pressure.",
                "Describe the old model compactly.",
                "Show how learning was practised or proven.",
                "Close with the mismatch without importing new examples.",
            ],
            "avoid": ["neighbor-topic expansion", "semantic change of no longer", "over-compressed history sentence"],
        },
        "claim_grounding_gap": {
            "movement": "local condition -> stated concern -> grounded contrast -> source closing terms",
            "paragraph_jobs": [
                "Keep the local condition and concern together.",
                "Ground the broad claim with source or approved proxy material.",
                "Preserve the source closing terms as the paragraph limit.",
            ],
            "avoid": ["invented evidence", "research tail", "new recommendation", "dropped closing list"],
        },
        "recommendation_process": {
            "movement": "need for change -> retained core -> added capabilities -> assessment/process detail",
            "paragraph_jobs": [
                "Keep the recommendation scoped.",
                "Group related capabilities naturally.",
                "Close on the process details without making a slogan.",
            ],
            "avoid": ["policy expansion", "catalogue of skills", "generic final goal"],
        },
        "turning_point_conclusion": {
            "movement": "choice point -> two paths -> rejected false goal -> actual goal",
            "paragraph_jobs": [
                "Keep the contrast between paths clear.",
                "Avoid adding new evidence.",
                "Close on the submitted goal.",
            ],
            "avoid": ["grand conclusion", "new advice", "extra future claim"],
        },
        "anchor_then_compact_development": {
            "movement": "concrete anchor -> grouped development -> scoped claim",
            "paragraph_jobs": [
                "Name the anchor before the broad claim.",
                "Group related details by role.",
                "Avoid a separate sentence for each finding.",
            ],
            "avoid": ["vague demonstrative start", "scanner-list rhythm", "unsupported broad claim"],
        },
        "packed_list_flow_rebuild": {
            "movement": "source claim -> grouped list role -> consequence",
            "paragraph_jobs": [
                "Identify what the list items do in the sentence.",
                "Keep only natural compact lists.",
                "Move consequence into its own clear beat.",
            ],
            "avoid": ["keyword dump", "over-compression", "same-shape short sentences"],
        },
        "claim_reason_consequence": {
            "movement": "claim -> reason -> consequence",
            "paragraph_jobs": [
                "Keep the source claim.",
                "State the source reason plainly.",
                "Close on the source consequence without adding a new topic.",
            ],
            "avoid": ["sentence-by-sentence repair", "new claim", "generic conclusion"],
        },
    }
    return routes.get(pattern_id, routes["claim_reason_consequence"])


def _tag_counts(plan: Plan) -> dict[str, int]:
    counts: dict[str, int] = {}
    for action in plan.actions:
        for tag in action.tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def _pattern_text(plan: Plan) -> str:
    action_text = " ".join(action.source_text for action in plan.actions if action.source_text.strip())
    context = plan.author_proxy_context if isinstance(plan.author_proxy_context, dict) else {}
    terms = context.get("context_terms")
    if isinstance(terms, list) and terms:
        return action_text + " " + " ".join(str(term) for term in terms if str(term).strip())
    return action_text
