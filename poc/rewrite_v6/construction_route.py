from __future__ import annotations

import re
from typing import Any

from .finding_pattern import classify_finding_pattern
from .plan import Plan
from .text import Paragraph, source_anchor_terms


def build_validated_construction_route(
    paragraph: Paragraph,
    plan: Plan,
    decision_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback = _fallback_route(paragraph, plan)
    route = decision_route if isinstance(decision_route, dict) else {}
    if not _valid_route(route):
        return fallback
    sanitized = {
        "route_id": _clean_text(route.get("route_id") or fallback["route_id"]),
        "movement": _clean_text(route.get("movement") or fallback["movement"]),
        "sentence_jobs": _sanitize_jobs(route.get("sentence_jobs"), fallback["sentence_jobs"]),
        "do_not_copy": _dedupe([*(_strings(route.get("do_not_copy"))), *fallback["do_not_copy"]])[:16],
        "route_rewrite_guidance": _dedupe([*(_strings(route.get("route_rewrite_guidance"))), *fallback["route_rewrite_guidance"]])[:10],
        "validation_rules": _dedupe([*(_strings(route.get("validation_rules"))), *fallback["validation_rules"]])[:16],
    }
    return sanitized if _valid_route(sanitized) else fallback


def _fallback_route(paragraph: Paragraph, plan: Plan) -> dict[str, Any]:
    pattern = classify_finding_pattern(plan)
    actions = [action for action in plan.actions if action.source_text.strip()]
    paragraph_jobs = pattern.get("paragraph_jobs") or ["Carry source meaning through a clearer paragraph beat."]
    job_count = len(paragraph_jobs)
    jobs = [
        {
            "job_id": f"beat_{index + 1}",
            "job": _clean_text(job),
            "source_basis": _source_basis(actions, paragraph, index, job_count),
            "must_use_meaning": _meaning_anchors(actions, paragraph, index, job_count),
            "must_not_use": [],
        }
        for index, job in enumerate(paragraph_jobs)
    ]
    return {
        "route_id": _clean_text(pattern.get("pattern_id") or "source_order_route"),
        "movement": _clean_text(pattern.get("movement") or "source claim -> reason -> consequence"),
        "sentence_jobs": jobs[:6],
        "do_not_copy": [],
        "route_rewrite_guidance": [],
        "validation_rules": [
            "Do not create one sentence per scanner finding.",
            "Do not copy risky source shapes.",
            "Preserve source modality and closing consequence.",
        ],
    }


def _source_basis(actions: list[Any], paragraph: Paragraph, index: int, job_count: int) -> list[str]:
    if actions:
        return [action.sentence_id for action in _action_slice(actions, index, job_count)]
    return [sentence.id for sentence in paragraph.sentences[:1]]


def _meaning_anchors(actions: list[Any], paragraph: Paragraph, index: int, job_count: int) -> list[str]:
    source = " ".join(action.source_text for action in _action_slice(actions, index, job_count)) if actions else paragraph.text
    return source_anchor_terms(source, term_limit=8, phrase_limit=2)


def _action_slice(actions: list[Any], index: int, job_count: int) -> list[Any]:
    if not actions:
        return []
    count = max(1, job_count)
    start = min(len(actions), (index * len(actions)) // count)
    end = min(len(actions), ((index + 1) * len(actions)) // count)
    if end <= start:
        end = min(len(actions), start + 1)
    return actions[start:end]


def _valid_route(route: dict[str, Any]) -> bool:
    jobs = route.get("sentence_jobs")
    return bool(route.get("movement") and isinstance(jobs, list) and len(jobs) >= 2)


def _sanitize_jobs(value: Any, fallback: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return fallback
    rows = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        job = _clean_text(item.get("job"))
        if not job:
            continue
        rows.append({
            "job_id": _clean_text(item.get("job_id") or f"job_{index + 1}")[:48],
            "job": job[:220],
            "source_basis": _strings(item.get("source_basis"))[:6],
            "must_use_meaning": _strings(item.get("must_use_meaning"))[:10],
            "must_not_use": _strings(item.get("must_not_use"))[:8],
        })
    return rows or fallback


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        rows = [value]
    elif isinstance(value, list):
        rows = value
    else:
        rows = []
    return [_clean_text(item) for item in rows if _clean_text(item)]


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows
