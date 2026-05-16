"""Convert scanner evidence into editorial repair briefs."""

from __future__ import annotations

import json
import os
from typing import Any

from llm.gateway import LLMGateway
from rewrite_v3.document_units import word_count

from .models import RepairBrief
from .validation import parse_json_object, sanitize_repair_brief


def _int_env(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)) or default)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


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
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V4_NORMALIZER_MAX_TOKENS", 3000, minimum=600, maximum=8000),
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


def tutor_repair_brief(group: Any, gateway: LLMGateway) -> RepairBrief:
    payload = {
        "task": "write_tutor_style_editorial_repair_brief",
        "paragraph": getattr(group, "source_text", ""),
        "paragraph_role_hint": _paragraph_role(group),
        "internal_scanner_evidence": scanner_evidence_for_group(group),
        "finding_glossary": {
            "ai_likelihood": "The overall prose pattern resembles broad machine-like overview writing. This is only an internal signal, not proof of authorship.",
            "ai_signal_score": "The paragraph has several combined texture signals that make it read too generated or template-like.",
            "predictability_score": "The sentence route and wording are easy to anticipate.",
            "unsafe_word_share": "Too much of the paragraph relies on broad, generic wording.",
            "rewrite_smoothness": "The paragraph may be too evenly polished or mechanically connected.",
        },
        "rules": [
            "Act like a writing tutor explaining the problem to a student, but output structured JSON only.",
            "Use scanner evidence to diagnose the writing pattern; do not expose raw scanner terms in the output.",
            "Use only excerpts that appear in the paragraph.",
            "Do not recommend new facts, new examples, personal anecdotes, slang, rare synonyms, or random errors.",
            "Do not suggest illustrative content that is absent from the paragraph; repair directions must describe a source-near writing move, not supply new subject matter.",
            "When asking for specificity, use only the paragraph's existing nouns, claims, and relationships.",
            "Source-near specificity means moving an existing source anchor earlier, qualifying an existing broad claim, or connecting two existing source claims more clearly.",
            "If the paragraph makes a broad unsupported claim, instruct the writer to narrow or connect that claim to existing source wording instead of adding a concrete example.",
            "Do not ask the writer to choose one narrow angle when the paragraph is a broad overview; preserve the paragraph's breadth and central claims.",
            "Do not ask the writer to remove a source sentence's central claim; each original sentence must remain represented even if the order changes.",
            "Never ask for a particular field, mechanism, place, period, event, person, organization, cultural practice, or industry unless the exact subject already appears in the paragraph.",
            "Do not use phrases like for example, such as, or e.g. in repair directions unless the phrase is quoting text already present in the paragraph.",
            "Prefer moves like: move an existing anchor earlier; replace an empty transition; connect adjacent source claims; qualify a broad sentence; group an existing list without adding list items.",
            "Give concrete repair directions that preserve meaning and paragraph role.",
            "Return only JSON with paragraph_role, tutor_diagnosis, student_explanation, source_examples, repair_assignment, repair_tasks, constraints, avoid, coverage_hint.",
        ],
        "response_schema": {
            "paragraph_role": "...",
            "tutor_diagnosis": "short explanation of why this paragraph reads weakly",
            "student_explanation": "student-facing explanation of the repeated writing pattern",
            "source_examples": [
                {
                    "excerpt": "exact phrase or sentence from paragraph",
                    "issue": "what feels too broad, predictable, list-like, or mechanical",
                    "repair_direction": "source-near writing move only; do not name absent subject matter",
                }
            ],
            "repair_assignment": "one concise source-near assignment for the generator; preserve broad scope when the paragraph is broad",
            "repair_tasks": ["..."],
            "constraints": ["..."],
            "avoid": ["..."],
            "coverage_hint": "local | paragraph | multi_paragraph",
        },
    }
    prompt = "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    response = gateway.chat(
        prompt,
        system="Return only valid JSON.",
        response_format={"type": "json_object"},
        temperature=0.15,
        top_p=0.8,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V4_NORMALIZER_MAX_TOKENS", 3000, minimum=600, maximum=8000),
    )
    data, diagnostics = parse_json_object(response.content)
    if data is None:
        return sanitize_repair_brief(
            normalizer="tutor_v0",
            paragraph_role=_paragraph_role(group),
            repair_tasks=[],
            constraints=[],
            avoid=[],
            parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
        )
    return sanitize_repair_brief(
        normalizer="tutor_v0",
        paragraph_role=data.get("paragraph_role") or _paragraph_role(group),
        tutor_diagnosis=data.get("tutor_diagnosis"),
        student_explanation=data.get("student_explanation"),
        source_examples=data.get("source_examples"),
        repair_assignment=data.get("repair_assignment"),
        repair_tasks=data.get("repair_tasks"),
        constraints=data.get("constraints"),
        avoid=data.get("avoid"),
        coverage_hint=data.get("coverage_hint"),
        parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
    )


def enrichment_repair_brief(group: Any, gateway: LLMGateway) -> RepairBrief:
    payload = {
        "task": "write_controlled_enrichment_repair_brief",
        "paragraph": getattr(group, "source_text", ""),
        "paragraph_role_hint": _paragraph_role(group),
        "internal_scanner_evidence": scanner_evidence_for_group(group),
        "finding_glossary": {
            "ai_likelihood": "The prose pattern reads like a broad generic overview. This is only an internal signal, not proof of authorship.",
            "ai_signal_score": "The paragraph has combined texture signals that make it read too template-like.",
            "predictability_score": "The sentence route and wording are easy to anticipate.",
            "unsafe_word_share": "Too much of the paragraph relies on broad wording that lacks grounding.",
        },
        "rules": [
            "Act like a writing tutor, but output structured JSON only.",
            "Create a controlled enrichment assignment only when source-preserving edits are likely to be too weak.",
            "Do not expose raw scanner terms in the output.",
            "Preserve all central source claims and the paragraph role.",
            "Allowed enrichment is limited to one or two short grounding phrases that explain or connect the existing claims.",
            "Do not introduce names, locations, organizations, events, dates, statistics, citations, quotes, or verifiable factual specifics absent from the paragraph.",
            "Do not suggest personal anecdotes, fake experience, slang, rare synonyms, or random errors.",
            "Do not turn a broad paragraph into one narrow example.",
            "If enrichment is needed, describe the type of general context to add, not a finished factual claim.",
            "Return only JSON with paragraph_role, tutor_diagnosis, student_explanation, source_examples, repair_assignment, repair_tasks, constraints, avoid, coverage_hint.",
        ],
        "response_schema": {
            "paragraph_role": "...",
            "tutor_diagnosis": "short explanation of why source-only repair may be weak",
            "student_explanation": "student-facing explanation of what grounding is missing",
            "source_examples": [
                {
                    "excerpt": "exact phrase or sentence from paragraph",
                    "issue": "what feels broad, ungrounded, or template-like",
                    "repair_direction": "bounded enrichment move; no absent proper nouns, dates, statistics, citations, or specific examples",
                }
            ],
            "repair_assignment": "one concise controlled-enrichment assignment for the generator",
            "repair_tasks": ["..."],
            "constraints": ["..."],
            "avoid": ["..."],
            "coverage_hint": "paragraph",
        },
    }
    prompt = "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    response = gateway.chat(
        prompt,
        system="Return only valid JSON.",
        response_format={"type": "json_object"},
        temperature=0.15,
        top_p=0.8,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V4_NORMALIZER_MAX_TOKENS", 3000, minimum=600, maximum=8000),
    )
    data, diagnostics = parse_json_object(response.content)
    if data is None:
        return sanitize_repair_brief(
            normalizer="enrichment_v0",
            paragraph_role=_paragraph_role(group),
            repair_tasks=[],
            constraints=[],
            avoid=[],
            parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
        )
    return sanitize_repair_brief(
        normalizer="enrichment_v0",
        paragraph_role=data.get("paragraph_role") or _paragraph_role(group),
        tutor_diagnosis=data.get("tutor_diagnosis"),
        student_explanation=data.get("student_explanation"),
        source_examples=data.get("source_examples"),
        repair_assignment=data.get("repair_assignment"),
        repair_tasks=data.get("repair_tasks"),
        constraints=data.get("constraints"),
        avoid=data.get("avoid"),
        coverage_hint=data.get("coverage_hint"),
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
