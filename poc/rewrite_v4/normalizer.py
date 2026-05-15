"""Convert scanner evidence into editorial repair briefs."""

from __future__ import annotations

import json
from typing import Any

from llm.gateway import LLMGateway
from rewrite_v3.document_units import word_count

from .models import RepairBrief
from .validation import parse_json_object, sanitize_repair_brief


def scanner_evidence_for_group(group: Any) -> dict[str, Any]:
    drivers: dict[str, dict[str, Any]] = {}
    sentence_ids: set[str] = set()
    target_ids: list[str] = []
    for target in getattr(group, "targets", ()) or ():
        if not isinstance(target, dict):
            continue
        target_id = str(target.get("target_id") or "")
        if target_id:
            target_ids.append(target_id)
        for sentence_id in target.get("sentence_ids") or []:
            if str(sentence_id):
                sentence_ids.add(str(sentence_id))
        for driver in target.get("dominant_drivers") or []:
            if not isinstance(driver, dict):
                continue
            key = str(driver.get("key") or "")
            if not key:
                continue
            score = driver.get("score")
            current = drivers.get(key)
            if current is None or _number(score) > _number(current.get("score")):
                drivers[key] = {"key": key, "score": score}
    return {
        "target_ids": target_ids,
        "sentence_ids": sorted(sentence_ids),
        "dominant_drivers": list(drivers.values()),
        "operation": getattr(group, "operation", ""),
        "word_count": word_count(getattr(group, "source_text", "")),
    }


def deterministic_repair_brief(group: Any) -> RepairBrief:
    evidence = scanner_evidence_for_group(group)
    keys = {str(row.get("key") or "") for row in evidence.get("dominant_drivers") or []}
    tasks: list[str] = []
    if "predictability_score" in keys:
        tasks.append("Vary the sentence route slightly so the paragraph does not move in an overly neat sequence.")
    if "unsafe_word_share" in keys:
        tasks.append("Replace broad or generic wording with clearer ordinary phrasing while keeping the same meaning.")
    if "ai_signal_score" in keys or "ai_likelihood" in keys:
        tasks.append("Make the paragraph read more like a careful human edit, with natural sentence linkage and less textbook-like phrasing.")
    if evidence.get("word_count", 0) > 70:
        tasks.append("Do not compress the paragraph; preserve the same amount of detail.")
    else:
        tasks.append("Keep the paragraph close to the original length and role.")
    return sanitize_repair_brief(
        normalizer="deterministic_v0",
        paragraph_role=_paragraph_role(group),
        repair_tasks=tasks,
        constraints=[
            "Preserve meaning.",
            "Do not add facts.",
            "Do not add examples or anecdotes.",
            "Keep one paragraph.",
            "Use a clear simple tone.",
            "Do not make it casual.",
        ],
        avoid=[
            "polished abstract summary",
            "word-by-word synonym swap",
            "overly casual phrasing",
        ],
    )


def llm_repair_brief(group: Any, gateway: LLMGateway) -> RepairBrief:
    payload = {
        "task": "convert_scanner_evidence_to_editorial_repair_task",
        "paragraph": getattr(group, "source_text", ""),
        "paragraph_role_hint": _paragraph_role(group),
        "internal_scanner_evidence": scanner_evidence_for_group(group),
        "rules": [
            "Translate scanner evidence into plain editorial repair tasks.",
            "Do not include words: AI, detector, scanner, likelihood, score, risk, bypass, evade.",
            "Do not ask for personal anecdotes, fake experience, slang, random errors, or new examples.",
            "Keep tasks small and actionable for a paragraph editor.",
            "Return only JSON with paragraph_role, repair_tasks, constraints, avoid.",
        ],
        "response_schema": {
            "paragraph_role": "...",
            "repair_tasks": ["..."],
            "constraints": ["..."],
            "avoid": ["..."],
        },
    }
    prompt = "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    response = gateway.chat(
        prompt,
        system="Return only valid JSON.",
        response_format={"type": "json_object"},
        temperature=0.1,
        top_p=0.8,
        max_tokens=900,
    )
    data, diagnostics = parse_json_object(response.content)
    if data is None:
        return sanitize_repair_brief(
            normalizer="llm_v0",
            paragraph_role=_paragraph_role(group),
            repair_tasks=[],
            constraints=[],
            avoid=[],
            parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
        )
    return sanitize_repair_brief(
        normalizer="llm_v0",
        paragraph_role=data.get("paragraph_role") or _paragraph_role(group),
        repair_tasks=data.get("repair_tasks"),
        constraints=data.get("constraints"),
        avoid=data.get("avoid"),
        parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
    )


def _paragraph_role(group: Any) -> str:
    unit_id = str(getattr(group, "unit_id", "") or "")
    return "opening/background framing" if unit_id == "p001" else "body paragraph"


def _number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
