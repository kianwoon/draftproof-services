"""Formula-gap candidate orchestration helpers.

This module keeps the product policy out of the large rewrite pipeline:
deterministic repair is only a probe, while the main candidate budget is
reserved for a small portfolio of formula-targeted LLM candidates.
"""

from __future__ import annotations

import json
import re
from typing import Any

from detect.turnitin_like import TURNITIN_LIKE_TARGET_AI_SCORE, turnitin_like_ai_profile_from_report


DEFAULT_DETERMINISTIC_PROBES = 2
DEFAULT_LLM_CANDIDATES = 5
DEFAULT_FINALIST_SCANS = 5
DEFAULT_TOTAL_SCAN_CAP = 10

PORTFOLIO_FAMILIES = (
    "STATISTICAL_TEXTURE_REBUILD",
    "SEMANTIC_VARIANCE_RESTRUCTURE",
    "HUMAN_ANCHOR_SUPPRESSION_GAIN",
    "HYBRID_TEXTURE_ANCHOR",
    "LOW_VALUE_COMPRESS_REMOVE",
)

_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})"
    r"(?:\s+(?:of|the|and|for|to|in|on|at|by|with|from|[A-Z][a-z]+|[A-Z]{2,})){1,7}\b"
)
_SINGLE_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-zA-Z0-9&.-]{2,}|[A-Z]{2,})\b")
_ENTITY_SKIP_PREFIXES = {
    "This",
    "That",
    "These",
    "Those",
    "Many",
    "Some",
    "Another",
    "One",
    "In",
    "At",
    "As",
    "But",
    "However",
    "Although",
    "Understanding",
    "Millions",
    "Innovation",
    "Universities",
    "Healthcare",
    "While",
    "Technology",
    "Throughout",
}
_ENTITY_SKIP_SINGLE = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "One",
    "Another",
    "Many",
    "Some",
    "Although",
    "However",
    "Despite",
    "Understanding",
    "Healthcare",
    "Technology",
    "In",
    "At",
    "As",
    "But",
    "It",
    "Its",
    "They",
    "Their",
    "Supporters",
    "Critics",
    "Throughout",
}


def named_entity_inventory(source_text: str, *, limit: int = 60) -> list[str]:
    """Extract visible named-entity anchors for prompt-side preservation."""

    seen: set[str] = set()
    entities: list[str] = []
    for match in _ENTITY_RE.finditer(str(source_text or "")):
        entity = " ".join(match.group(0).split()).strip(" ,.;:!?")
        if not entity:
            continue
        first = entity.split()[0]
        words = entity.split()
        if first in _ENTITY_SKIP_PREFIXES:
            continue
        if words[-1].lower() in {"of", "the", "and", "in", "with", "from"}:
            continue
        if entity.lower() in seen:
            continue
        seen.add(entity.lower())
        entities.append(entity)
        if len(entities) >= int(limit):
            return entities
    for match in _SINGLE_ENTITY_RE.finditer(str(source_text or "")):
        entity = match.group(0).strip(" ,.;:!?")
        if entity in _ENTITY_SKIP_SINGLE:
            continue
        if entity.lower() in seen:
            continue
        seen.add(entity.lower())
        entities.append(entity)
        if len(entities) >= int(limit):
            return entities
    return entities


def formula_gap_plan(report_dict: dict | None) -> dict[str, Any]:
    profile = turnitin_like_ai_profile_from_report(report_dict or {})
    weighted = profile.get("weighted_components") if isinstance(profile.get("weighted_components"), dict) else {}
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    score = float(profile.get("score") or 0.0)
    target = float(profile.get("target_score") or TURNITIN_LIKE_TARGET_AI_SCORE)
    gap = max(0.0, score - target)
    suppression = float(profile.get("human_anchor_suppression") or 0.0)
    remaining = []
    for driver, contribution in weighted.items():
        if not isinstance(contribution, (int, float)):
            continue
        remaining.append({
            "driver": driver,
            "value": components.get(driver),
            "weighted_contribution": round(float(contribution), 3),
        })
    remaining.sort(key=lambda row: float(row.get("weighted_contribution") or 0.0), reverse=True)
    remaining.append({
        "driver": "human_anchor_suppression",
        "value": round(suppression, 3),
        "target_direction": "increase",
        "available_suppression_headroom": round(max(0.0, 45.0 - suppression), 3),
        "weighted_contribution": round(-suppression, 3),
    })
    return {
        "version": "formula_gap_candidate_orchestrator_v1",
        "score": round(score, 3),
        "target_score": target,
        "target_gap": round(gap, 3),
        "target_met": bool(profile.get("target_met")),
        "weighted_components": weighted,
        "components": components,
        "human_anchor_suppression": round(suppression, 3),
        "suppression_headroom": round(max(0.0, 45.0 - suppression), 3),
        "remaining_weighted_drivers": remaining,
        "dominant_drivers": [
            row["driver"]
            for row in remaining
            if row["driver"] != "human_anchor_suppression"
        ][:4],
    }


def budget_contract(
    *,
    deterministic_probes: int = DEFAULT_DETERMINISTIC_PROBES,
    llm_candidates: int = DEFAULT_LLM_CANDIDATES,
    finalist_scans: int = DEFAULT_FINALIST_SCANS,
    total_scan_cap: int = DEFAULT_TOTAL_SCAN_CAP,
) -> dict[str, int]:
    return {
        "deterministic_probe_scans": max(0, int(deterministic_probes)),
        "llm_candidate_calls": max(0, int(llm_candidates)),
        "finalist_scans": max(0, int(finalist_scans)),
        "total_scan_cap": max(1, int(total_scan_cap)),
    }


def portfolio_families(limit: int = DEFAULT_LLM_CANDIDATES) -> list[str]:
    return list(PORTFOLIO_FAMILIES[: max(0, int(limit))])


def formula_gap_candidate_prompt(
    source_text: str,
    report_dict: dict | None,
    family: str,
    *,
    protected_anchors: list[dict] | None = None,
) -> str:
    plan = formula_gap_plan(report_dict)
    family_instructions = {
        "STATISTICAL_TEXTURE_REBUILD": (
            "Change statistical texture: reduce model-like cadence, predictable openings, "
            "balanced clause routes, and rewrite smoothness. Preserve facts."
        ),
        "SEMANTIC_VARIANCE_RESTRUCTURE": (
            "Change paragraph jobs and reasoning order. Reduce repeated claim-explain-conclude "
            "flow and semantic uniformity without adding facts."
        ),
        "HUMAN_ANCHOR_SUPPRESSION_GAIN": (
            "Add bounded implied process reasoning, limitations, and author judgement that follow "
            "from the submitted content. Do not invent lived events or evidence."
        ),
        "HYBRID_TEXTURE_ANCHOR": (
            "Combine texture reduction with bounded implied reasoning. Move both positive AI "
            "drivers and human-anchor suppression."
        ),
        "LOW_VALUE_COMPRESS_REMOVE": (
            "Compress or remove broad low-information blocks while preserving required claims, "
            "facts, names, dates, citations, and argument continuity."
        ),
    }.get(family, "Reduce the weighted formula score while preserving facts.")
    schema = {
        "strategy": family,
        "targeted_drivers": [],
        "changed_blocks": [],
        "fact_inventory_preserved": True,
        "core_claims_preserved_or_merged": True,
        "protected_anchors_preserved": True,
        "unsupported_new_facts": False,
        "candidate_text": "complete rewritten candidate here",
    }
    return (
        "DraftProof FORMULA_GAP_PORTFOLIO_CANDIDATE.\n"
        "Objective: produce one candidate that lowers the shared Turnitin-like AI score below 20 if possible.\n"
        "Selection is based on a full rescan. Do not optimize a single raw signal if total weighted score gets worse.\n\n"
        f"Portfolio family: {family}\n"
        f"Family instruction: {family_instructions}\n\n"
        f"Current formula plan:\n{json.dumps(plan, ensure_ascii=False)[:3600]}\n\n"
        "Hard constraints:\n"
        "- Preserve all named entities, dates, numbers, citations, quotes, protected anchors, and core factual claims.\n"
        "- Every protected anchor listed below must remain visible in the candidate unless it is explicitly duplicated elsewhere in equivalent form.\n"
        "- Build the candidate by editing the source document, not by drafting from memory. Copy unchanged blocks exactly.\n"
        "- Prefer 2-5 high-impact block/sentence changes over rewriting every paragraph.\n"
        "- If a fact/example is hard to improve safely, keep the original wording for that fact/example.\n"
        "- Do not add fake people, fake dates, fake sources, unsupported evidence, or fabricated lived experience.\n"
        "- Do not use personal voice as a default operation.\n"
        "- You may change structure, paragraph order, pacing, and explanation density if facts remain intact.\n"
        "- Avoid generic polished transitions and balanced essay rhythm.\n"
        "- Keep argument continuity and readable academic tone.\n\n"
        f"Protected anchors:\n{json.dumps(protected_anchors or [], ensure_ascii=False)[:2600]}\n\n"
        "Return only valid JSON matching this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "SOURCE DOCUMENT:\n"
        f"<TARGET_DOCUMENT>\n{source_text}\n</TARGET_DOCUMENT>"
    )


def extract_candidate_payload(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = str(raw or "").strip()
    if not text:
        return None, "empty_response"
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            text = match.group(0)
    try:
        payload = json.loads(text)
    except Exception as exc:  # pragma: no cover - exact JSON exception is not important to callers
        return None, f"invalid_json {exc}"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    candidate_text = str(payload.get("candidate_text") or "").strip()
    if not candidate_text:
        return None, "missing_candidate_text"
    if payload.get("unsupported_new_facts") is True:
        return None, "unsupported_new_facts_declared"
    for key in ("fact_inventory_preserved", "core_claims_preserved_or_merged", "protected_anchors_preserved"):
        if payload.get(key) is False:
            return None, f"{key}_false"
    return payload, ""
