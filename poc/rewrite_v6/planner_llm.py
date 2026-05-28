from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Protocol

from .construction_route import build_validated_construction_route
from .finding_pattern import classify_finding_pattern
from .json_io import parse_json
from .paragraph_planner import build_paragraph_planner_brief
from .plan import Plan
from .prose_repair_rules import (
    consolidated_prose_repair_rules,
    planner_agent_profile,
    risky_source_copy_phrases,
    risky_source_rewrite_guidance,
    risky_source_shape_guidance,
    risky_source_terms,
)
from .scan import Finding
from .text import Paragraph
from .writer_prompt_compact import compact_document_signal_contracts


class PlannerClient(Protocol):
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        ...


def run_planner_llm(paragraph: Paragraph, plan: Plan, findings: list[Finding], *, client: PlannerClient) -> Plan:
    prompt = build_planner_prompt(paragraph, plan, findings)
    last_error = "planner_failed"
    for _ in range(2):
        try:
            decision = _request_planner_decision(prompt, client)
            unsafe_reason = _unsafe_planner_reason(decision)
            if unsafe_reason:
                return _with_source_order_planner(paragraph, plan, unsafe_reason)
            decision["contract_gaps"] = _planner_contract_gaps(decision, findings, plan)
            return _merge_decision(paragraph, plan, decision)
        except Exception as exc:
            last_error = type(exc).__name__
    return _with_source_order_planner(paragraph, plan, last_error)


def _request_planner_decision(prompt: str, client: PlannerClient) -> dict[str, Any]:
    response = client.chat(
        prompt,
        system=(
            "Return valid JSON only with a planner_decision object. Do not write replacement prose. "
            "planner_decision.flow_plan must be a non-empty list of paragraph movement steps."
        ),
        temperature=0.1,
        top_p=0.75,
        max_tokens=None,
        response_format={"type": "json_object"},
        app_label="Planner",
    )
    raw = getattr(response, "raw_content", "") or response.content
    return _planner_decision(parse_json(raw))


def build_planner_prompt(paragraph: Paragraph, plan: Plan, findings: list[Finding]) -> str:
    dense_plan = plan.paragraph_strategy.get("dense_paragraph_plan", {}) if isinstance(plan.paragraph_strategy, dict) else {}
    grounding = plan.paragraph_strategy.get("author_proxy_grounding", {}) if isinstance(plan.paragraph_strategy, dict) else {}
    finding_pattern = classify_finding_pattern(plan)
    payload = {
        "task": "v6_curated_writer_brief_plan",
        "planner_role": "Decide the paragraph route and discard irrelevant or harmful context before the writer sees it.",
        "planner_agent_profile": planner_agent_profile(paragraph.text),
        "paragraph": {
            "paragraph_id": paragraph.id,
            "source_text": paragraph.text,
            "sentence_count": len(paragraph.sentences),
        },
        "scanner_findings": [
            {
                "sentence_id": finding.sentence_id,
                "tags": finding.tags,
                "severity": finding.severity,
                "text": finding.evidence.get("text"),
            }
            for finding in findings
        ],
        "source_sentence_contracts": [
            {
                "sentence_id": action.sentence_id,
                "source_text": action.source_text,
                "finding_tags": action.tags,
                "preserve_terms": action.preserve_terms,
            }
            for action in plan.actions
        ],
        "author_proxy_decision_rubric": {
            "request_author_proxy_when": [
                "scanner_findings include author_anchor_gap or unsupported_claim_gap",
                "the affected claim cannot be grounded by selected-paragraph source_sentence_contracts",
                "the missing bridge needs submitted neighbor context to avoid a detached broad claim",
            ],
            "do_not_request_author_proxy_when": [
                "findings are only packed_list, predictable_start, sentence_overload, paragraph_rhythm, or transition_stack",
                "the vague pointer has a clear antecedent inside the selected paragraph",
                "the paragraph can be fixed by reorganising selected-paragraph terms",
                "neighbor context would introduce a new topic or stronger claim",
            ],
            "required_request_fields": [
                "required must be true only for a real content or anchor gap, not a style problem",
                "reason must name the triggering scanner finding tags and why local source terms are insufficient",
                "target_sentence_ids must list the specific finding sentence ids that need Author-proxy grounding",
            ],
        },
        "finding_pattern_route": finding_pattern,
        "global_prose_repair_rules": consolidated_prose_repair_rules(),
        "risky_source_shape_policy": {
            "risky_shapes_in_this_paragraph": risky_source_shape_guidance(paragraph.text),
            "risky_copy_phrases_in_this_paragraph": risky_source_copy_phrases(paragraph.text),
            "route_rewrite_guidance": risky_source_rewrite_guidance(paragraph.text),
            "risky_terms_not_wording_requirements": sorted(risky_source_terms(paragraph.text)),
            "planner_rule": (
                "Preserve the source meaning behind risky shapes, but do not put risky terms into flow_plan.must_include "
                "or coverage_terms as exact wording requirements."
            ),
        },
        "route_skeleton": {
            "paragraph_strategy": {
                "repair_unit": plan.paragraph_strategy.get("repair_unit"),
                "target_sentence_range": plan.paragraph_strategy.get("target_sentence_range"),
                "dense_paragraph_plan": {
                    "dominant_flow_problem": dense_plan.get("dominant_flow_problem", ""),
                    "finding_interpretation_rule": dense_plan.get("finding_interpretation_rule", ""),
                },
                "author_proxy_grounding": {
                    "required": bool(grounding.get("required")) if isinstance(grounding, dict) else False,
                    "reason": grounding.get("reason", "") if isinstance(grounding, dict) else "",
                },
                "author_proxy_pack": plan.paragraph_strategy.get("author_proxy_pack", {}),
            },
            "document_signal_contracts": compact_document_signal_contracts(plan.ai_safe_route.get("document_signal_contracts", [])),
        },
        "output_contract": {
            "root_key": "planner_decision",
            "required_keys": [
                "paragraph_problem",
            "flow_plan",
            "validated_construction_route",
                "paragraph_route",
                "coverage_terms",
                "author_proxy_request",
                "do_not_copy_route",
                "route_rewrite_guidance",
            ],
            "field_rules": [
                "paragraph_problem must name the actual paragraph problem in this paragraph.",
            "flow_plan is mandatory and must contain 2-4 paragraph movement steps using real submitted sentence ids.",
            "validated_construction_route must contain movement, sentence_jobs, do_not_copy, and validation_rules.",
            "validated_construction_route.sentence_jobs must be ordered route jobs, not final prose.",
            "coverage_terms must contain source meaning that must survive, not risky wording that must be copied.",
            "flow_plan must preserve the closing source beat and its preserve_terms when the closing sentence has findings.",
                "paragraph_route must be a concrete route name or sentence-level movement description for this paragraph.",
                "author_proxy_request must be an object with required, reason, target_sentence_ids, and finding_basis.",
                "do_not_copy_route must list exact risky shapes from this paragraph, not generic placeholders.",
                "route_rewrite_guidance must list route instructions separately from copy-blocked phrases.",
            ],
            "required_output_shape": {
                "planner_decision": {
                    "paragraph_problem": "string explaining the paragraph-level problem",
                    "paragraph_route": "short route description",
                    "flow_plan": [
                        {
                            "step_id": "fp001",
                            "function": "paragraph movement step, not final prose",
                            "source_basis": ["submitted sentence id"],
                            "must_include": ["meaning anchors only, not risky wording"],
                        }
                    ],
                    "validated_construction_route": {
                        "route_id": "route identifier",
                        "movement": "ordered paragraph movement",
                        "sentence_jobs": [
                            {
                                "job_id": "stable route job id",
                                "job": "what this paragraph beat must accomplish",
                                "source_basis": ["submitted sentence id"],
                                "must_use_meaning": ["meaning anchors only"],
                                "must_not_use": ["exact risky source wording"],
                            }
                        ],
                        "do_not_copy": ["exact risky source wording"],
                        "validation_rules": ["specific handoff check the writer must pass"],
                    },
                    "do_not_copy_route": ["exact risky source wording"],
                    "route_rewrite_guidance": ["route instruction, not copy-blocked wording"],
                    "coverage_terms": ["source meaning that must survive"],
                    "author_proxy_request": {
                        "required": False,
                        "reason": "",
                        "target_sentence_ids": [],
                        "finding_basis": [],
                    },
                }
            },
            "forbidden_output_values": [
                "one plain-language sentence",
                "paragraph movement job",
                "submitted sentence id",
                "source-supported terms",
                "specific malformed or machine route",
                "positive route the writer must follow",
                "machine route shapes to avoid",
            ],
        },
        "rules": [
            "Do not write replacement paragraph prose or final sentence shapes.",
            "Return only the fields in output_contract.required_keys.",
            "Use only claims, actors, objects, and relations present in the selected paragraph unless author_proxy_request is required.",
            "Build 2-4 flow_plan beats; do not plan one job per scanner finding or source sentence.",
            "Use validated_construction_route.sentence_jobs only for ordered route logic.",
            "Put risky exact wording into do_not_copy_route and validated_construction_route.do_not_copy.",
            "Keep risky wording out of flow_plan.must_include, coverage_terms, and sentence_jobs.must_use_meaning.",
            "Preserve source meaning, modality, and closing consequence.",
            "Request Author-proxy only for a real grounding gap that local source terms cannot solve.",
            "If author_proxy_pack exists, integrate its usable bridge as grounding in flow_plan and validated_construction_route, not as prose.",
        ],
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _planner_decision(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    decision = payload.get("planner_decision", payload)
    return decision if isinstance(decision, dict) else {}


def _placeholder_decision(decision: dict[str, Any]) -> bool:
    markers = (
        "one plain-language sentence",
        "paragraph movement job",
        "submitted sentence id",
        "source-supported terms",
        "specific malformed or machine route",
        "positive route the writer must follow",
        "machine route shapes to avoid",
    )
    return any(marker in value for value in _decision_strings(decision) for marker in markers)


def _unsafe_planner_reason(decision: dict[str, Any]) -> str:
    if _placeholder_decision(decision):
        return "planner_placeholder_output"
    lowered = " ".join(_decision_strings(decision)).casefold()
    bad_expansion_patterns = (
        r"\badd\b[^.]{0,80}\bnew\b[^.]{0,80}\bexample\b",
        r"\binsert\b[^.]{0,80}\bnew\b[^.]{0,80}\bclaim\b",
        r"\bexpand\b[^.]{0,80}\bbroad claim\b",
        r"\bextend beyond\b[^.]{0,80}\bsource\b",
        r"\bconcrete illustration\b[^.]{0,80}\bnot present\b",
    )
    if any(re.search(pattern, lowered) for pattern in bad_expansion_patterns):
        return "planner_unbounded_expansion"
    return ""


def _with_source_order_planner(paragraph: Paragraph, plan: Plan, reason: str) -> Plan:
    route = dict(plan.ai_safe_route)
    fallback = build_paragraph_planner_brief(plan)
    construction_route = _sanitize_construction_route_anchors(
        build_validated_construction_route(paragraph, plan),
        plan,
    )
    coverage_beats = _coverage_beats(plan, construction_route)
    route["llm_planner_decision"] = {
        "status": "fallback",
        "paragraph_problem": fallback.get("paragraph_diagnosis", reason),
        "paragraph_route": fallback.get("paragraph_route", ""),
        "flow_plan": _recipe_rows(fallback.get("flow_plan")),
        "validated_construction_route": construction_route,
        "do_not_copy_route": _copy_blockers(plan, construction_route, {}),
        "route_rewrite_guidance": _route_rewrite_guidance(plan, construction_route, {}),
        "coverage_beats": coverage_beats,
        "coverage_terms": _coverage_terms(plan),
        "raw_merged_route_alignment": False,
        "coverage_beat_gaps": _coverage_beat_gaps(coverage_beats, construction_route, plan),
        "author_proxy_request": _author_proxy_request({}, plan),
    }
    return replace(plan, ai_safe_route=route)


def _decision_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        rows: list[str] = []
        for item in value.values():
            rows.extend(_decision_strings(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_decision_strings(item))
        return rows
    return []


def _merge_decision(paragraph: Paragraph, plan: Plan, decision: dict[str, Any]) -> Plan:
    route = dict(plan.ai_safe_route)
    contract_gaps = _string_rows(decision.get("contract_gaps"))
    unsafe_contracts = _unsafe_contract_gaps(contract_gaps)
    dense_plan = plan.paragraph_strategy.get("dense_paragraph_plan", {})
    if not isinstance(dense_plan, dict):
        dense_plan = {}
    paragraph_fallback = build_paragraph_planner_brief(plan)
    flow_plan = _sanitize_flow_plan(
        _recipe_rows(decision.get("flow_plan")) or _recipe_rows(paragraph_fallback.get("flow_plan")),
        plan,
    )
    fallback_flow_plan = _recipe_rows(paragraph_fallback.get("flow_plan"))
    paragraph_problem = decision.get("paragraph_problem") or paragraph_fallback.get("paragraph_diagnosis", "")
    paragraph_route = decision.get("paragraph_route") or paragraph_fallback.get("paragraph_route", "")
    construction_route = build_validated_construction_route(
        paragraph,
        plan,
        decision.get("validated_construction_route") if isinstance(decision.get("validated_construction_route"), dict) else None,
    )
    construction_route = _sanitize_construction_route_anchors(construction_route, plan)
    if unsafe_contracts:
        flow_plan = fallback_flow_plan
        construction_route = _sanitize_construction_route_anchors(
            build_validated_construction_route(paragraph, plan, None),
            plan,
        )
    coverage_beats = _coverage_beats(plan, construction_route, decision)
    coverage_beat_gaps = _coverage_beat_gaps(coverage_beats, construction_route, plan)
    if coverage_beat_gaps:
        fallback_route = _sanitize_construction_route_anchors(
            build_validated_construction_route(paragraph, plan, None),
            plan,
        )
        fallback_beats = _coverage_beats(plan, fallback_route, decision)
        fallback_gaps = _coverage_beat_gaps(fallback_beats, fallback_route, plan)
        if len(fallback_gaps) <= len(coverage_beat_gaps):
            flow_plan = _sanitize_flow_plan(fallback_flow_plan, plan)
            construction_route = fallback_route
            coverage_beats = fallback_beats
            coverage_beat_gaps = fallback_gaps
    request = _author_proxy_request(decision, plan)
    strategy = dict(plan.paragraph_strategy)
    grounding = dict(strategy.get("author_proxy_grounding", {})) if isinstance(strategy.get("author_proxy_grounding"), dict) else {}
    grounding["required"] = bool(request.get("required"))
    grounding["reason"] = str(request.get("reason") or grounding.get("reason") or "")
    strategy["author_proxy_grounding"] = grounding
    route["llm_planner_decision"] = {
        "status": _planner_status(contract_gaps),
        "paragraph_problem": paragraph_problem,
        "paragraph_route": paragraph_route,
        "flow_plan": flow_plan,
        "validated_construction_route": construction_route,
        "do_not_copy_route": _copy_blockers(plan, construction_route, decision),
        "route_rewrite_guidance": _route_rewrite_guidance(plan, construction_route, decision),
        "coverage_beats": coverage_beats,
        "coverage_terms": _coverage_terms(plan, decision),
        "raw_merged_route_alignment": _raw_merged_route_alignment(decision.get("validated_construction_route"), construction_route),
        "coverage_beat_gaps": coverage_beat_gaps,
        "author_proxy_request": request,
    }
    return replace(plan, paragraph_strategy=strategy, ai_safe_route=route)


def _planner_status(contract_gaps: list[str]) -> str:
    if not contract_gaps:
        return "ok"
    if any("unknown source sentence id" in gap for gap in contract_gaps):
        return "invalid_source_basis"
    if any("fewer than" in gap or "missing validated_construction_route" in gap for gap in contract_gaps):
        return "invalid_route_shape"
    return "degraded_contract_gaps"


def _author_proxy_request(decision: dict[str, Any], plan: Plan) -> dict[str, Any]:
    request = decision.get("author_proxy_request") if isinstance(decision.get("author_proxy_request"), dict) else {}
    grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    grounding_required = bool(grounding.get("required")) if isinstance(grounding, dict) else False
    required = request.get("required") if "required" in request else grounding_required
    target_ids = request.get("target_sentence_ids")
    basis = request.get("finding_basis")
    return {
        "required": bool(required),
        "reason": str(request.get("reason") or (grounding.get("reason") if isinstance(grounding, dict) else "") or "").strip(),
        "target_sentence_ids": [str(item).strip() for item in target_ids if str(item).strip()][:4] if isinstance(target_ids, list) else [],
        "finding_basis": [str(item).strip() for item in basis if str(item).strip()][:6] if isinstance(basis, list) else [],
    }


def _copy_blockers(plan: Plan, construction_route: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    source_text = _plan_source_text(plan)
    rows = [
        *_string_rows(construction_route.get("do_not_copy")),
        *_string_rows(decision.get("do_not_copy_route")),
        *risky_source_copy_phrases(source_text),
    ]
    return [
        phrase
        for phrase in _dedupe_strings(rows)
        if _copy_blocker_phrase(phrase)
    ][:24]


def _route_rewrite_guidance(plan: Plan, construction_route: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    instruction_rows = [
        row
        for row in _string_rows(decision.get("do_not_copy_route"))
        if not _copy_blocker_phrase(row)
    ]
    rows = [
        *_string_rows(construction_route.get("route_rewrite_guidance")),
        *_string_rows(decision.get("route_rewrite_guidance")),
        *risky_source_rewrite_guidance(_plan_source_text(plan)),
        *instruction_rows,
    ]
    return _dedupe_strings([_guidance_text(row) for row in rows if _guidance_text(row)])[:16]


def _copy_blocker_phrase(value: str) -> bool:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return False
    lowered = text.casefold()
    if lowered.startswith(("do not ", "preserve ", "replace ", "show ", "close ")):
        return False
    if ";" in text or len(text.split()) > 8:
        return False
    return True


def _guidance_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if text.casefold().startswith("do not "):
        return "Preserve source meaning through the planned route without copying blocked source wording."
    return text


def _sanitize_flow_plan(flow_plan: list[dict[str, Any]], plan: Plan) -> list[dict[str, Any]]:
    source_text = _plan_source_text(plan)
    rows: list[dict[str, Any]] = []
    for row in flow_plan:
        updated = dict(row)
        must_include = updated.get("must_include")
        if isinstance(must_include, list):
            updated["must_include"] = _safe_anchor_values(must_include, source_text)
        rows.append(updated)
    return rows


def _sanitize_construction_route_anchors(construction_route: dict[str, Any], plan: Plan) -> dict[str, Any]:
    source_text = _plan_source_text(plan)
    updated = dict(construction_route)
    jobs = construction_route.get("sentence_jobs")
    if isinstance(jobs, list):
        sanitized_jobs: list[dict[str, Any]] = []
        for job in jobs:
            if not isinstance(job, dict):
                continue
            sanitized = dict(job)
            sanitized["must_use_meaning"] = _safe_anchor_values(_string_rows(job.get("must_use_meaning")), source_text)
            sanitized_jobs.append(sanitized)
        updated["sentence_jobs"] = sanitized_jobs
    return updated


def _plan_source_text(plan: Plan) -> str:
    return " ".join(str(action.source_text or "") for action in plan.actions)


def _coverage_terms(plan: Plan, decision: dict[str, Any] | None = None) -> list[str]:
    source_text = _plan_source_text(plan)
    return _safe_anchor_values([
        str(term)
        for action in plan.actions
        for term in getattr(action, "preserve_terms", [])
        ] + _string_rows((decision or {}).get("coverage_terms")) + _author_proxy_coverage_terms(plan),
        source_text,
    )[:40]


def _coverage_beats(plan: Plan, construction_route: dict[str, Any], decision: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    source_by_id = {action.sentence_id: action.source_text for action in plan.actions}
    rows: list[dict[str, Any]] = []
    jobs = construction_route.get("sentence_jobs") if isinstance(construction_route, dict) else []
    if not isinstance(jobs, list):
        jobs = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            continue
        basis = [
            str(source_id)
            for source_id in job.get("source_basis", [])
            if str(source_id) in source_by_id
        ]
        sources = [source_by_id[source_id] for source_id in basis]
        meaning = _coverage_meaning(sources)
        if not meaning:
            meaning = _coverage_meaning(_string_rows(job.get("must_use_meaning")))
        if not meaning:
            continue
        must_not_copy = _local_must_not_copy(job, sources)
        meaning = _safe_coverage_meaning(meaning, must_not_copy)
        rows.append({
            "beat_id": f"cb{index:03d}",
            "route_job_id": str(job.get("job_id") or f"job_{index}"),
            "source_basis": basis,
            "meaning": meaning,
            "must_cover": _short_must_cover(job, meaning),
            "must_not_copy": must_not_copy,
        })
    return _dedupe_coverage_beats(rows)[:10]


def _short_must_cover(job: dict[str, Any], meaning: str) -> list[str]:
    anchors = _relation_anchors(str(job.get("job_id") or ""), _string_rows(job.get("must_use_meaning")))
    if not anchors:
        anchors = _short_anchor_phrases(meaning)
    return [anchor for anchor in _dedupe_strings(anchors) if len(anchor.split()) <= 9][:6]


def _relation_anchors(job_id: str, anchors: list[str]) -> list[str]:
    return anchors


def _local_must_not_copy(job: dict[str, Any], sources: list[str]) -> list[str]:
    configured = _string_rows(job.get("must_not_use"))
    text = " ".join(sources)
    lowered = text.casefold()
    phrases = [
        phrase for phrase in configured
        if phrase.casefold() in lowered
    ]
    return _dedupe_strings([*phrases, *risky_source_copy_phrases(text)])[:8]


def _safe_coverage_meaning(meaning: str, must_not_copy: list[str]) -> str:
    text = str(meaning or "").strip()
    for phrase in must_not_copy:
        text = re.sub(re.escape(str(phrase)), "[copy-blocked source wording]", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _short_anchor_phrases(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", str(text or ""))
    if len(words) <= 9:
        return [str(text).strip()] if str(text).strip() else []
    return [" ".join(words[:5]), " ".join(words[-5:])]


def _safe_anchor_values(values: list[str], source_text: str) -> list[str]:
    blocked = risky_source_copy_phrases(source_text)
    risky = risky_source_terms(source_text)
    rows: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip(" .,;:")
        if not text:
            continue
        if text.casefold() in risky or _contains_blocked_phrase(text, blocked):
            continue
        if _sentence_like_anchor(text):
            rows.extend(
                anchor for anchor in _short_anchor_phrases(text)
                if not _contains_blocked_phrase(anchor, blocked)
            )
            continue
        rows.append(text)
    return _dedupe_strings(rows)


def _contains_blocked_phrase(text: str, blocked: list[str]) -> bool:
    lowered = str(text or "").casefold()
    return any(phrase.casefold() in lowered for phrase in blocked if phrase)


def _sentence_like_anchor(text: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'’-]*", str(text or ""))
    return len(words) > 9 or ("," in text and len(words) > 6)


def _coverage_meaning(sources: list[str]) -> str:
    phrases = [_clean_coverage_phrase(source) for source in sources]
    phrases = [phrase for phrase in phrases if phrase]
    return "; ".join(phrases[:3])


def _clean_coverage_phrase(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" .")
    value = re.sub(r"^(?:However|But|Also|Moreover|Furthermore|Therefore|Thus|So),?\s+", "", value, flags=re.I)
    value = value[:1].lower() + value[1:] if value[:1].isupper() and not value.startswith("AI") else value
    return value.strip()


def _dedupe_coverage_beats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        meaning = re.sub(r"\s+", " ", str(row.get("meaning") or "")).strip()
        route_job_id = str(row.get("route_job_id") or "").strip()
        key = f"{route_job_id}:{meaning}".casefold()
        if not meaning or key in seen:
            continue
        updated = dict(row)
        updated["meaning"] = meaning
        kept.append(updated)
        seen.add(key)
    return kept


def _raw_merged_route_alignment(raw_route: Any, merged_route: dict[str, Any]) -> bool:
    raw_text = " ".join(_decision_strings(raw_route)).casefold()
    merged_text = " ".join(_decision_strings(merged_route)).casefold()
    if not raw_text or not merged_text:
        return False
    return bool(set(_plain_route_words(raw_text)) & set(_plain_route_words(merged_text)))


def _coverage_beat_gaps(
    coverage_beats: list[dict[str, Any]],
    construction_route: dict[str, Any],
    plan: Plan,
) -> list[str]:
    gaps: list[str] = []
    valid_ids = {str(action.sentence_id) for action in plan.actions if str(action.sentence_id)}
    covered_ids: set[str] = set()
    route_job_ids = {
        str(job.get("job_id"))
        for job in construction_route.get("sentence_jobs", [])
        if isinstance(job, dict) and str(job.get("job_id") or "").strip()
    }
    beat_job_ids = set()
    for beat in coverage_beats:
        if not isinstance(beat, dict):
            continue
        beat_id = str(beat.get("beat_id") or "unknown")
        job_id = str(beat.get("route_job_id") or "")
        if job_id:
            beat_job_ids.add(job_id)
        basis = beat.get("source_basis")
        if not isinstance(basis, list) or not basis:
            gaps.append(f"coverage beat missing source_basis: {beat_id}")
            continue
        for source_id in basis:
            sid = str(source_id)
            if sid not in valid_ids:
                gaps.append(f"coverage beat uses unknown source sentence id: {sid}")
            else:
                covered_ids.add(sid)
        for item in _string_rows(beat.get("must_cover")):
            if len(item.split()) > 9:
                gaps.append(f"coverage beat must_cover is sentence-shaped: {beat_id}")
        meaning = str(beat.get("meaning") or "").casefold()
        for phrase in _string_rows(beat.get("must_not_copy")):
            if phrase.casefold() in meaning:
                gaps.append(f"{beat_id} meaning contains must_not_copy phrase: {phrase}")
    missing_jobs = sorted(route_job_ids - beat_job_ids)
    for job_id in missing_jobs:
        gaps.append(f"route job missing coverage beat: {job_id}")
    required_ids = {str(action.sentence_id) for action in plan.actions if str(action.source_text or "").strip()}
    for source_id in sorted(required_ids - covered_ids):
        gaps.append(f"source sentence missing from coverage beats: {source_id}")
    return gaps[:8]


def _plain_route_words(text: str) -> list[str]:
    stop = {"the", "and", "that", "this", "with", "from", "into", "route", "sentence", "job"}
    return [word for word in re.findall(r"[a-z][a-z'-]{3,}", text) if word not in stop]


def _author_proxy_coverage_terms(plan: Plan) -> list[str]:
    pack = plan.paragraph_strategy.get("author_proxy_pack", {}) if isinstance(plan.paragraph_strategy, dict) else {}
    bridges = pack.get("usable_bridges") if isinstance(pack, dict) else []
    if not isinstance(bridges, list):
        return []
    return [
        str(bridge.get("usable_bridge") or "").strip()
        for bridge in bridges
        if isinstance(bridge, dict) and str(bridge.get("usable_bridge") or "").strip()
    ][:3]


def _dedupe_strings(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows


def _recipe_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _string_rows(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(row) for row in value if str(row).strip()]


def _unsafe_contract_gaps(gaps: list[str]) -> bool:
    return bool(gaps)


def _planner_contract_gaps(decision: dict[str, Any], findings: list[Finding], plan: Plan | None = None) -> list[str]:
    route_gaps = _planner_route_gaps(decision, plan) if plan is not None else []
    contracts = _recipe_rows(decision.get("finding_contracts"))
    reasons: list[str] = []
    if not contracts:
        return route_gaps
    if len(contracts) < len(findings):
        reasons.append("missing finding_contract rows for one or more scanner findings")
    by_sentence: dict[str, list[dict[str, Any]]] = {}
    for contract in contracts:
        sentence_id = str(contract.get("source_sentence_id") or "").strip()
        if sentence_id:
            by_sentence.setdefault(sentence_id, []).append(contract)
        safe = str(contract.get("safe_rebuild_shape") or "")
        if "<" in safe or ">" in safe:
            reasons.append(f"{sentence_id or 'unknown'} safe_rebuild_shape uses placeholder brackets")
        if ";" in safe:
            reasons.append(f"{sentence_id or 'unknown'} safe_rebuild_shape uses semicolon punctuation that the writer contract forbids")
        if safe.count(",") >= 2:
            reasons.append(f"{sentence_id or 'unknown'} safe_rebuild_shape keeps a comma-list route instead of a safer construction route")
        opener = _forbidden_sentence_opener(safe)
        if opener:
            reasons.append(f"{sentence_id or 'unknown'} safe_rebuild_shape uses forbidden sentence opener '{opener}'")
        bridge_term = _unsubmitted_bridge_term(safe, contract)
        if bridge_term:
            reasons.append(f"{sentence_id or 'unknown'} safe_rebuild_shape introduces unsubmitted bridge term '{bridge_term}'")
        planning_label = _planning_label(safe)
        if planning_label:
            reasons.append(f"{sentence_id or 'unknown'} safe_rebuild_shape contains planning label '{planning_label}'")
    finding_text = {finding.sentence_id: str(finding.evidence.get("text") or "") for finding in findings}
    for finding in findings:
        if not (set(finding.tags) & {"broad_claim", "context_anchor_gap", "predictable_start"}):
            continue
        source = finding_text.get(finding.sentence_id, "")
        for contract in by_sentence.get(finding.sentence_id, []):
            safe = str(contract.get("safe_rebuild_shape") or "")
            copied = _copied_phrase(source, safe, min_words=4)
            if copied:
                reasons.append(
                    f"{finding.sentence_id} safe_rebuild_shape copies risky source phrase '{copied}' instead of making a new route"
                )
    return [*route_gaps, *reasons][:8]


def _planner_route_gaps(decision: dict[str, Any], plan: Plan) -> list[str]:
    gaps: list[str] = []
    flow = _recipe_rows(decision.get("flow_plan"))
    route = decision.get("validated_construction_route")
    jobs = route.get("sentence_jobs") if isinstance(route, dict) else []
    source_ids = {str(action.sentence_id) for action in plan.actions if str(action.sentence_id)}
    risky_terms = risky_source_terms(_plan_source_text(plan))

    if not str(decision.get("paragraph_problem") or "").strip():
        gaps.append("missing paragraph_problem")
    if not str(decision.get("paragraph_route") or "").strip():
        gaps.append("missing paragraph_route")
    if len(flow) < 2:
        gaps.append("flow_plan has fewer than 2 movement steps")
    if len(flow) > 4:
        gaps.append("flow_plan has more than 4 movement steps")
    if not isinstance(route, dict):
        gaps.append("missing validated_construction_route")
        jobs = []
    if not isinstance(jobs, list) or len(jobs) < 2:
        gaps.append("validated_construction_route has fewer than 2 sentence_jobs")

    gaps.extend(_basis_gaps("flow_plan", flow, source_ids))
    gaps.extend(_basis_gaps("sentence_job", jobs if isinstance(jobs, list) else [], source_ids))
    gaps.extend(_risky_term_gaps("flow_plan.must_include", flow, "must_include", risky_terms))
    gaps.extend(_risky_term_gaps("sentence_job.must_use_meaning", jobs if isinstance(jobs, list) else [], "must_use_meaning", risky_terms))
    gaps.extend(_risky_term_gaps("coverage_terms", [{"coverage_terms": _string_rows(decision.get("coverage_terms"))}], "coverage_terms", risky_terms))
    return gaps[:8]


def _basis_gaps(label: str, rows: list[dict[str, Any]], source_ids: set[str]) -> list[str]:
    gaps: list[str] = []
    for index, row in enumerate(rows, start=1):
        basis = row.get("source_basis") if isinstance(row, dict) else None
        if not isinstance(basis, list) or not basis:
            gaps.append(f"{label} {index} missing source_basis")
            continue
        unknown = [str(item) for item in basis if str(item) not in source_ids]
        if unknown:
            gaps.append(f"{label} {index} uses unknown source sentence id {unknown[0]}")
    return gaps


def _risky_term_gaps(label: str, rows: list[dict[str, Any]], key: str, risky_terms: set[str]) -> list[str]:
    if not risky_terms:
        return []
    gaps: list[str] = []
    for index, row in enumerate(rows, start=1):
        values = row.get(key) if isinstance(row, dict) else None
        if not isinstance(values, list):
            continue
        copied = [str(value) for value in values if str(value).casefold() in risky_terms]
        if copied:
            gaps.append(f"{label} {index} copies risky source term {copied[0]}")
    return gaps


def _has_curated_route(decision: dict[str, Any]) -> bool:
    """Accept the small Planner contract when it gives route logic.

    Planner is not a hidden writer. It should not need finding_contract rows or
    safe sentence shapes when it has supplied paragraph movement and a validated
    construction route.
    """
    route = decision.get("validated_construction_route")
    jobs = route.get("sentence_jobs") if isinstance(route, dict) else []
    flow_plan = decision.get("flow_plan")
    return bool(
        str(decision.get("paragraph_route") or "").strip()
        and isinstance(flow_plan, list)
        and len([row for row in flow_plan if isinstance(row, dict)]) >= 2
        and isinstance(jobs, list)
        and len([row for row in jobs if isinstance(row, dict)]) >= 2
    )


def _copied_phrase(source: str, candidate: str, *, min_words: int) -> str:
    source_words = _plain_words(source)
    candidate_text = " ".join(_plain_words(candidate))
    if len(source_words) < min_words or not candidate_text:
        return ""
    for size in range(min(7, len(source_words)), min_words - 1, -1):
        for index in range(0, len(source_words) - size + 1):
            phrase_words = source_words[index:index + size]
            phrase = " ".join(phrase_words)
            if phrase in candidate_text:
                return phrase
    return ""


def _plain_words(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", str(text or ""))
        if token.casefold() not in {"the", "and", "that", "this", "with", "from", "through"}
    ]


def _forbidden_sentence_opener(text: str) -> str:
    for sentence in re.split(r"(?<=[.!?])\s+", str(text or "").strip()):
        match = re.match(r"^[\"'“”‘’]*([A-Za-z]+)\b", sentence.strip())
        if match and match.group(1).casefold() in {"and", "but", "or", "which", "where", "that", "as", "this", "these", "those"}:
            return match.group(1)
    return ""


def _unsubmitted_bridge_term(text: str, contract: dict[str, Any]) -> str:
    allowed = {
        token.casefold()
        for term in contract.get("coverage_terms", [])
        for token in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", str(term))
    }
    risky = {"other", "various", "broader", "critical", "complex", "significant", "important"}
    for token in re.findall(r"[A-Za-z][A-Za-z'’-]{2,}", str(text or "")):
        key = token.casefold()
        if key in risky and key not in allowed:
            return token
    return ""


def _planning_label(text: str) -> str:
    labels = {"relation", "beat", "anchor", "route", "contract", "source term", "writer"}
    lowered = str(text or "").casefold()
    for label in labels:
        if label in lowered:
            return label
    return ""
