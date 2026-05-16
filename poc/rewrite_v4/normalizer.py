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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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
        "structure": _source_structure(getattr(group, "source_text", "")),
    }


def deterministic_repair_brief(group: Any) -> RepairBrief:
    evidence = scanner_evidence_for_group(group)
    keys = {str(row.get("key") or "") for row in evidence.get("dominant_drivers") or []}
    role = _paragraph_role(group)
    tasks: list[str] = []
    if role == "opening/background framing":
        tasks.append("Let an existing concrete course, unit, or workplace anchor carry the opening before broad background claims.")
        tasks.append("Make the concrete anchor the subject of the opening route rather than keeping a generic importance-of statement.")
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
        paragraph_role=role,
        mitigation_strategy=_deterministic_mitigation_strategy(group, evidence=evidence, role=role),
        repair_tasks=tasks,
        constraints=[
            "Preserve meaning.",
            "Do not add facts.",
            "Do not add examples or anecdotes.",
            "Keep one paragraph.",
            "Use a clear simple tone.",
            "Do not make it casual.",
            "Preserve the writer's student voice; do not globally polish grammar or upgrade the register.",
        ],
        avoid=[
            "polished abstract summary",
            "word-by-word synonym swap",
            "overly casual phrasing",
            "professional copyediting voice",
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
            "Derive one typed mitigation_strategy that tells the rewriter what strategy to use.",
            "Do not include words: detector, scanner, likelihood, score, risk, bypass, evade.",
            "Paint the current paragraph route and a better source-preserving route for the writer.",
            "The better route must reorder existing claims only; it must not invent new subject matter.",
            "Do not ask for personal anecdotes, fake experience, slang, random errors, or new examples.",
            "Keep tasks small and actionable for a paragraph editor.",
            "Return only JSON with paragraph_role, mitigation_strategy, repair_tasks, constraints, avoid.",
        ],
        "response_schema": {
            "paragraph_role": "...",
            "mitigation_strategy": {
                "scope": "local_span | paragraph | multi_block_section | document_cluster",
                "strategy_id": "one short snake_case strategy name",
                "primary_problem": "one sentence",
                "rewrite_depth": "light | moderate | broad_but_source_preserving",
                "candidate_count_hint": "1 | 2 | 3",
                "strategy_steps": [
                    {
                        "op": "abstract_to_source_anchor | claim_bridge_repair | reaction_reason_link | list_to_process_repair",
                        "target": "source-local target area",
                        "instruction": "one executable edit instruction",
                        "must_preserve": ["..."],
                        "avoid": ["..."],
                    }
                ],
                "current_route": ["what the current paragraph does now, in order"],
                "better_route": ["better source-preserving route, in order"],
                "route_moves": ["specific move from current_route to better_route"],
                "route_must_preserve": ["source claims or anchors that must survive route changes"],
                "route_forbidden": ["old route pattern the writer must not keep"],
                "target_zones": ["..."],
                "required_moves": ["..."],
                "forbidden_moves": ["..."],
                "must_preserve_claims": ["..."],
                "success_checks": ["..."],
            },
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
    data, diagnostics = _parse_or_repair_json_object(
        response=response,
        gateway=gateway,
        response_schema=payload["response_schema"],
    )
    if data is None:
        role = _paragraph_role(group)
        return sanitize_repair_brief(
            normalizer="llm_v0",
            paragraph_role=role,
            mitigation_strategy=_merge_strategy_defaults(group, {}, role=role),
            repair_tasks=[],
            constraints=[],
            avoid=[],
            parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
        )
    role = data.get("paragraph_role") or _paragraph_role(group)
    return sanitize_repair_brief(
        normalizer="llm_v0",
        paragraph_role=role,
        mitigation_strategy=_merge_strategy_defaults(group, data.get("mitigation_strategy"), role=role),
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
            "Paint the current paragraph route and a better source-preserving route for the writer.",
            "The better route must reorder existing claims only; it must not invent new subject matter.",
            "Return only JSON with paragraph_role, mitigation_strategy, tutor_diagnosis, student_explanation, source_examples, repair_assignment, repair_tasks, constraints, avoid, coverage_hint.",
        ],
        "response_schema": {
            "paragraph_role": "...",
            "mitigation_strategy": {
                "scope": "local_span | paragraph | multi_block_section | document_cluster",
                "strategy_id": "one short snake_case strategy name",
                "primary_problem": "one sentence",
                "rewrite_depth": "light | moderate | broad_but_source_preserving",
                "candidate_count_hint": "1 | 2 | 3",
                "strategy_steps": [
                    {
                        "op": "abstract_to_source_anchor | claim_bridge_repair | reaction_reason_link | list_to_process_repair",
                        "target": "source-local target area",
                        "instruction": "one executable edit instruction",
                        "must_preserve": ["..."],
                        "avoid": ["..."],
                    }
                ],
                "current_route": ["what the current paragraph does now, in order"],
                "better_route": ["better source-preserving route, in order"],
                "route_moves": ["specific move from current_route to better_route"],
                "route_must_preserve": ["source claims or anchors that must survive route changes"],
                "route_forbidden": ["old route pattern the writer must not keep"],
                "target_zones": ["..."],
                "required_moves": ["..."],
                "forbidden_moves": ["..."],
                "must_preserve_claims": ["..."],
                "success_checks": ["..."],
            },
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
    data, diagnostics = _parse_or_repair_json_object(
        response=response,
        gateway=gateway,
        response_schema=payload["response_schema"],
    )
    if data is None:
        role = _paragraph_role(group)
        return sanitize_repair_brief(
            normalizer="tutor_v0",
            paragraph_role=role,
            mitigation_strategy=_merge_strategy_defaults(group, {}, role=role),
            repair_tasks=[],
            constraints=[],
            avoid=[],
            parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
        )
    role = data.get("paragraph_role") or _paragraph_role(group)
    return sanitize_repair_brief(
        normalizer="tutor_v0",
        paragraph_role=role,
        mitigation_strategy=_merge_strategy_defaults(group, data.get("mitigation_strategy"), role=role),
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
            "Paint the current paragraph route and a better source-preserving route for the writer.",
            "The better route must reorder existing claims only; it must not invent new subject matter.",
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
            "mitigation_strategy": {
                "scope": "local_span | paragraph | multi_block_section | document_cluster",
                "strategy_id": "one short snake_case strategy name",
                "primary_problem": "one sentence",
                "rewrite_depth": "light | moderate | broad_but_source_preserving",
                "candidate_count_hint": "1 | 2 | 3",
                "strategy_steps": [
                    {
                        "op": "abstract_to_source_anchor | claim_bridge_repair | reaction_reason_link | list_to_process_repair",
                        "target": "source-local target area",
                        "instruction": "one executable edit instruction",
                        "must_preserve": ["..."],
                        "avoid": ["..."],
                    }
                ],
                "current_route": ["what the current paragraph does now, in order"],
                "better_route": ["better source-preserving route, in order"],
                "route_moves": ["specific move from current_route to better_route"],
                "route_must_preserve": ["source claims or anchors that must survive route changes"],
                "route_forbidden": ["old route pattern the writer must not keep"],
                "target_zones": ["..."],
                "required_moves": ["..."],
                "forbidden_moves": ["..."],
                "must_preserve_claims": ["..."],
                "success_checks": ["..."],
            },
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
    data, diagnostics = _parse_or_repair_json_object(
        response=response,
        gateway=gateway,
        response_schema=payload["response_schema"],
    )
    if data is None:
        role = _paragraph_role(group)
        return sanitize_repair_brief(
            normalizer="enrichment_v0",
            paragraph_role=role,
            mitigation_strategy=_merge_strategy_defaults(group, {}, role=role),
            repair_tasks=[],
            constraints=[],
            avoid=[],
            parse_diagnostics={**diagnostics, "model": response.model, "provider": response.raw.get("provider")},
        )
    role = data.get("paragraph_role") or _paragraph_role(group)
    return sanitize_repair_brief(
        normalizer="enrichment_v0",
        paragraph_role=role,
        mitigation_strategy=_merge_strategy_defaults(group, data.get("mitigation_strategy"), role=role),
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


def _merge_strategy_defaults(group: Any, strategy: Any, *, role: str | None = None) -> dict[str, Any]:
    evidence = scanner_evidence_for_group(group)
    base = _deterministic_mitigation_strategy(group, evidence=evidence, role=role or _paragraph_role(group))
    incoming = strategy if isinstance(strategy, dict) else {}
    merged: dict[str, Any] = {}

    for key in ("scope", "strategy_id", "primary_problem", "rewrite_depth"):
        incoming_value = _clean_strategy_scalar(incoming.get(key))
        merged[key] = incoming_value or base.get(key)

    if base.get("scope") == "multi_block_section":
        merged["scope"] = "multi_block_section"

    merged["candidate_count_hint"] = _stricter_candidate_hint(
        incoming.get("candidate_count_hint"),
        base.get("candidate_count_hint"),
    )

    for key in (
        "target_zones",
        "current_route",
        "better_route",
        "route_moves",
        "route_must_preserve",
        "route_forbidden",
        "required_moves",
        "forbidden_moves",
        "must_preserve_claims",
        "success_checks",
    ):
        merged[key] = _merge_strategy_list(base.get(key), incoming.get(key), limit=8)

    merged["strategy_steps"] = _merge_strategy_steps(base.get("strategy_steps"), incoming.get("strategy_steps"), limit=5)
    return {key: value for key, value in merged.items() if value}


def _clean_strategy_scalar(value: Any) -> str:
    return " ".join(str(value or "").split())[:180]


def _merge_strategy_list(base_items: Any, incoming_items: Any, *, limit: int) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for source in (base_items, incoming_items):
        if not isinstance(source, list):
            continue
        for item in source:
            text = " ".join(str(item or "").split())
            key = text.casefold()
            if not text or key in seen:
                continue
            rows.append(text[:260])
            seen.add(key)
            if len(rows) >= limit:
                return rows
    return rows


def _merge_strategy_steps(base_steps: Any, incoming_steps: Any, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in (base_steps, incoming_steps):
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            op = _clean_strategy_scalar(item.get("op"))[:80]
            target = _clean_strategy_scalar(item.get("target"))[:120]
            instruction = _clean_strategy_scalar(item.get("instruction"))[:240]
            if not op or not target or not instruction:
                continue
            key = (op.casefold(), target.casefold())
            if key in seen:
                continue
            rows.append({
                "op": op,
                "target": target,
                "instruction": instruction,
                "must_preserve": _merge_strategy_list(item.get("must_preserve"), [], limit=4),
                "avoid": _merge_strategy_list(item.get("avoid"), [], limit=4),
            })
            seen.add(key)
            if len(rows) >= limit:
                return rows
    return rows


def _stricter_candidate_hint(incoming: Any, base: Any) -> str:
    hints: list[int] = []
    for value in (incoming, base):
        try:
            parsed = int(str(value or "").strip())
        except ValueError:
            continue
        if parsed > 0:
            hints.append(parsed)
    if not hints:
        return "2"
    return str(max(1, min(hints)))


def _deterministic_mitigation_strategy(group: Any, *, evidence: dict[str, Any], role: str) -> dict[str, Any]:
    structure = evidence.get("structure") if isinstance(evidence.get("structure"), dict) else {}
    multi_block = int(structure.get("block_count") or 1) > 1 or int(structure.get("nonempty_line_count") or 1) > 1
    driver_keys = {str(row.get("key") or "") for row in evidence.get("dominant_drivers") or [] if isinstance(row, dict)}
    target_zones: list[str] = []
    required_moves: list[str] = []
    forbidden_moves: list[str] = [
        "add absent facts, examples, names, dates, numbers, or citations",
        "upgrade the writer into a polished professional voice",
        "drop source claims to get a smoother paragraph",
    ]
    must_preserve: list[str] = [
        "source viewpoint",
        "protected anchors and citations",
        "central claim of each source sentence",
    ]
    must_preserve.extend(_source_claim_preserves(getattr(group, "source_text", ""), limit=4))
    if multi_block:
        scope = "multi_block_section"
        forbidden_moves.extend([
            "compress the section into only the opening block",
            "remove existing section headings",
        ])
        required_moves.append("preserve the section structure while repairing only weak routes and links")
        target_zones.append("section route and adjacent block transitions")
    else:
        scope = "paragraph"
        required_moves.append("keep the paragraph shape while repairing weak sentence routes")
        target_zones.append("paragraph route")

    if role == "opening/background framing":
        strategy_id = "anchor_first_route_repair"
        target_zones.append("opening/background frame")
        required_moves.extend([
            "start from an existing concrete course, unit, workplace, or source anchor when available",
            "connect broad context back to the existing anchor instead of using a generic importance frame",
        ])
        forbidden_moves.extend([
            "generic importance-of opening",
            "delete the original broad field claim when moving the anchor earlier",
        ])
    elif "unsafe_word_share" in driver_keys:
        strategy_id = "generic_claim_narrowing"
        required_moves.append("narrow broad wording by connecting it to existing source terms")
    elif "predictability_score" in driver_keys:
        strategy_id = "sentence_route_repair"
        required_moves.append("change predictable sentence routes without changing the argument")
    else:
        strategy_id = "source_near_texture_repair"
        required_moves.append("make small source-near texture repairs without adding new material")

    if "ai_signal_score" in driver_keys or "ai_likelihood" in driver_keys:
        required_moves.append("make links between existing claims feel more manually edited and less template-like")
    if "predictability_score" in driver_keys:
        required_moves.append("vary one or two predictable routes instead of replacing the whole unit")

    word_total = int(evidence.get("word_count") or 0)
    rewrite_depth = "broad_but_source_preserving" if word_total >= 250 else "moderate"
    route_plan = _deterministic_route_plan(group, role=role)
    return {
        "scope": scope,
        "strategy_id": strategy_id,
        "primary_problem": "the unit has a formulaic route and weak source-grounded linkage",
        "rewrite_depth": rewrite_depth,
        "candidate_count_hint": "1" if word_total >= 350 else "2",
        "target_zones": target_zones[:5],
        "current_route": route_plan["current_route"],
        "better_route": route_plan["better_route"],
        "route_moves": route_plan["route_moves"],
        "route_must_preserve": route_plan["route_must_preserve"],
        "route_forbidden": route_plan["route_forbidden"],
        "strategy_steps": _deterministic_strategy_steps(group, role=role, multi_block=multi_block),
        "required_moves": required_moves[:8],
        "forbidden_moves": forbidden_moves[:8],
        "must_preserve_claims": must_preserve[:6],
        "success_checks": [
            "same source blocks represented",
            "protected anchors preserved",
            "central source claims preserved",
            "no unsupported facts added",
            "measured texture signals do not worsen",
        ],
    }


def _deterministic_route_plan(group: Any, *, role: str) -> dict[str, list[str]]:
    source = str(getattr(group, "source_text", "") or "")
    claims = _source_claim_preserves(source, limit=6)
    if role != "opening/background framing":
        return {
            "current_route": claims[:5],
            "better_route": [
                "source-specific context",
                "actual difficulty or pressure already in the unit",
                "claim response using existing paragraph terms",
                "source-supported conclusion",
            ],
            "route_moves": [
                "Start from the source-specific context rather than a broad abstract claim.",
                "Move the concrete difficulty before the broad response claim where the source supports it.",
                "Keep citations and protected anchors attached to their original claims.",
            ],
            "route_must_preserve": claims[:5],
            "route_forbidden": [
                "drop source claims to make a smoother route",
                "replace the route with a generic summary",
            ],
        }
    return {
        "current_route": [
            "broad field claim",
            "AI/information-era practical-skills claim",
            "course name as delayed anchor",
            "inclusive design benefit",
            "citation support",
        ],
        "better_route": [
            "course or unit context",
            "actual learning difficulty already described",
            "AI/social-media shortcut pressure as part of that difficulty",
            "why inclusive design matters for that course context",
            "supported academic claim and citation",
        ],
        "route_moves": [
            "Move the course or unit context before the broad inclusive-design claim.",
            "Use the student difficulty with practical skill learning as the reason the route matters.",
            "Keep the AI/information-era or social-media shortcut pressure, but place it as part of the learning difficulty.",
            "Attach the inclusive-design benefit and citation to the response claim.",
        ],
        "route_must_preserve": claims[:6],
        "route_forbidden": [
            "broad claim first",
            "this is particularly noticeable bridge",
            "as a result generic benefit bridge",
            "drop AI/information-era or social-media shortcut pressure",
        ],
    }


def _deterministic_strategy_steps(group: Any, *, role: str, multi_block: bool) -> list[dict[str, Any]]:
    source = str(getattr(group, "source_text", "") or "")
    opening_claims = _source_claim_preserves(source, limit=3)
    steps: list[dict[str, Any]] = []
    if role == "opening/background framing":
        steps.append({
            "op": "abstract_to_source_anchor",
            "target": "opening/background frame",
            "instruction": "Start by saying what the existing course or unit shows, not that the topic is important.",
            "must_preserve": [
                "broad field claim",
                "source viewpoint",
                "existing citations",
                *opening_claims[:1],
            ],
            "avoid": [
                "generic importance-of opening",
                "becoming increasingly important route",
                "important because route",
                "this course shows why the topic matters route",
            ],
        })
        steps.append({
            "op": "claim_bridge_repair",
            "target": "broad context to source anchor transition",
            "instruction": "Connect broad context to the existing course or unit through the practical skill problem already described.",
            "must_preserve": [
                "original broad context claim",
                "course or unit anchor",
                "practical skill learning context",
                *opening_claims[1:3],
            ],
            "avoid": [
                "different-from-traditional-methods as a standalone bridge",
                "this is particularly noticeable bridge",
                "technology has changed everything bridge",
                "broad shift without a practical-skill link",
            ],
        })
    if "pretend" in source.casefold() or "untalented" in source.casefold():
        steps.append({
            "op": "reaction_reason_link",
            "target": "learner response cluster",
            "instruction": "Preserve each learner reaction and keep the reason a student hides confusion or pretends to understand.",
            "must_preserve": [
                "quiet or avoids asking questions",
                "pretends to understand",
                "fear of being judged or labelled untalented",
                "overconfidence or guesswork",
            ],
            "avoid": [
                "delete a learner reaction",
                "turn the cluster into a generic confidence sentence",
            ],
        })
    if "repeated practice" in source.casefold() or "feedback" in source.casefold():
        steps.append({
            "op": "list_to_process_repair",
            "target": "practice and correction sentence",
            "instruction": "Write practice, guided correction, feedback, and mistake correction as the sequence by which the skill forms.",
            "must_preserve": [
                "practice",
                "guided corrections",
                "feedback",
                "mistake correction",
            ],
            "avoid": [
                "compress the process into 'support'",
                "replace the learning process with a polished summary",
                "turn the process into a generic learning outcome",
            ],
        })
    if multi_block:
        steps.append({
            "op": "structure_preservation",
            "target": "multi-block section",
            "instruction": "Keep all original source blocks represented while editing only weak routes and links.",
            "must_preserve": [
                "existing headings",
                "all source blocks",
            ],
            "avoid": [
                "compress section into opening only",
                "remove section heading",
            ],
        })
    return steps[:5]


def _source_claim_preserves(source_text: Any, *, limit: int) -> list[str]:
    claims: list[str] = []
    for sentence in _source_sentences(str(source_text or "")):
        text = " ".join(sentence.split())
        if len(text.split()) < 8:
            continue
        claims.append(text[:220])
        if len(claims) >= limit:
            break
    return claims


def _source_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in str(text or ""):
        current.append(char)
        if char in ".?!":
            sentence = "".join(current).strip()
            current = []
            if sentence:
                sentences.append(sentence)
    remainder = "".join(current).strip()
    if remainder:
        sentences.append(remainder)
    return sentences


def _source_structure(source_text: Any) -> dict[str, Any]:
    lines = str(source_text or "").splitlines()
    block_count = 0
    in_block = False
    nonempty_line_count = 0
    for line in lines:
        if line.strip():
            nonempty_line_count += 1
            if not in_block:
                block_count += 1
                in_block = True
        else:
            in_block = False
    return {
        "block_count": max(1, block_count),
        "nonempty_line_count": max(1, nonempty_line_count),
    }


def _paragraph_role(group: Any) -> str:
    unit_id = str(getattr(group, "unit_id", "") or "")
    return "opening/background framing" if unit_id == "p001" else "body paragraph"


def _parse_or_repair_json_object(
    *,
    response: Any,
    gateway: LLMGateway,
    response_schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    raw_completion = response.raw_content or response.content
    data, diagnostics = parse_json_object(raw_completion)
    if data is not None or diagnostics.get("status") != "json_parse_failed":
        return data, diagnostics
    if not _bool_env("DRAFTPROOF_REWRITE_V4_REPAIR_MALFORMED_JSON", True):
        return data, diagnostics
    repair_payload = {
        "task": "repair_malformed_normalizer_json",
        "invalid_completion": str(raw_completion or ""),
        "rules": [
            "Do not add new analysis.",
            "Only convert the invalid completion into valid JSON.",
            "Escape straight double quote characters inside string values.",
            "Return only the fields requested by response_schema.",
        ],
        "response_schema": response_schema,
    }
    repair_response = gateway.chat(
        "Return valid JSON only.\n" + json.dumps(repair_payload, ensure_ascii=False, indent=2),
        system="Return only valid JSON.",
        response_format={"type": "json_object"},
        temperature=0.0,
        top_p=0.5,
        max_tokens=_int_env("DRAFTPROOF_REWRITE_V4_JSON_REPAIR_MAX_TOKENS", 8000, minimum=800, maximum=16000),
    )
    repaired, repair_diagnostics = parse_json_object(repair_response.raw_content or repair_response.content)
    if repaired is not None:
        return repaired, {
            **repair_diagnostics,
            "status": "ok_after_json_repair",
            "first_parse": diagnostics,
            "repair_model": repair_response.model,
            "repair_provider": repair_response.raw.get("provider"),
        }
    return None, {
        **diagnostics,
        "repair_attempt": {
            **repair_diagnostics,
            "model": repair_response.model,
            "provider": repair_response.raw.get("provider"),
        },
    }


def _number(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0
