from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Protocol

from .json_io import parse_json
from .plan import Plan
from .scan import Finding
from .text import Paragraph


class PlannerClient(Protocol):
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        ...


def run_planner_llm(paragraph: Paragraph, plan: Plan, findings: list[Finding], *, client: PlannerClient) -> Plan:
    try:
        prompt = build_planner_prompt(paragraph, plan, findings)
        response = client.chat(
            prompt,
            system="Return valid JSON only with a planner_decision object. Do not write replacement prose.",
            temperature=0.1,
            top_p=0.75,
            max_tokens=None,
            response_format={"type": "json_object"},
        )
        raw = getattr(response, "raw_content", "") or response.content
        decision = _planner_decision(parse_json(raw))
        decision["contract_gaps"] = _planner_contract_gaps(decision, findings)
    except Exception as exc:
        return _with_planner_status(plan, {"status": "fallback", "reason": type(exc).__name__})
    return _merge_decision(plan, decision)


def build_planner_prompt(paragraph: Paragraph, plan: Plan, findings: list[Finding]) -> str:
    payload = {
        "task": "v6_paragraph_safe_route_plan",
        "planner_role": "Convert scanner findings into executable construction recipes for the writer.",
        "paragraph": {
            "paragraph_id": paragraph.id,
            "source_text": paragraph.text,
            "sentence_count": len(paragraph.sentences),
            "sentences": [
                {"sentence_id": sentence.id, "text": sentence.text, "word_count": sentence.word_count}
                for sentence in paragraph.sentences
            ],
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
        "deterministic_route_skeleton": {
            "paragraph_strategy": plan.paragraph_strategy,
            "coverage_beats": _compact_rows(plan.ai_safe_route.get("coverage_beats", []), limit=16),
            "construction_recipes": _compact_rows(plan.ai_safe_route.get("construction_recipes", []), limit=16),
            "golden_route": plan.ai_safe_route.get("golden_route", {}),
        },
        "required_decision": {
            "paragraph_route": "positive route the writer must follow",
            "finding_contracts": [
                {
                    "finding_id": "sentence id plus tag or stable finding label",
                    "source_sentence_id": "scanner sentence id",
                    "finding_tags": ["scanner finding tag"],
                    "unsafe_original_shape": "specific shape in the source that causes the finding",
                    "safe_rebuild_shape": "positive replacement shape using actual submitted source terms, not final prose",
                    "writer_must_do": ["concrete action the writer must execute"],
                    "writer_must_not_do": ["specific shortcut or copied route to avoid"],
                    "coverage_terms": ["source-supported terms that must survive"],
                }
            ],
            "paragraph_blueprint": [
                {
                    "step_id": "b001",
                    "function": "what this paragraph beat must do",
                    "source_basis": ["submitted sentence id or source term"],
                    "must_include": ["source-supported term or relation"],
                    "must_avoid_shape": ["unsafe shape this beat must not copy"],
                    "safe_sentence_shape": "placeholder shape only, not final prose",
                }
            ],
            "finding_recipe_overrides": [
                {
                    "source_sentence_id": "sentence id",
                    "safe_route": "finding-specific construction route",
                    "build_steps": ["concrete writer step"],
                    "positive_pattern": "placeholder pattern only, not replacement prose",
                }
            ],
            "author_proxy_plan": "how to use reviewable bridges when anchors are missing",
            "do_not_copy_route": ["same opener", "same list rhythm", "same closure shape"],
        },
        "rules": [
            "Do not write replacement paragraph prose.",
            "Do not hardcode domain-specific starts or examples.",
            "Return one finding_contract for every scanner_findings row.",
            "Every finding_contract must target the exact finding tags and sentence id from scanner_findings.",
            "Do not use placeholder-only safe shapes such as <anchor>, <relation>, or <claim>.",
            "Each safe_rebuild_shape must include actual submitted source terms from the source sentence or neighboring paragraph context.",
            "Do not put planning labels such as relation, beat, anchor, route, contract, source term, or writer in safe_rebuild_shape.",
            "Write safe_rebuild_shape as an ordinary prose skeleton the writer can turn into final text without copying planning language.",
            "For context_anchor_gap or broad_claim findings, name the concrete antecedent and the scoped relation the writer should use.",
            "For broad_claim findings, replace the broad predicate with a scoped partial-relation shape instead of reusing the flagged predicate wording.",
            "For closure findings, split continuity and limitation into separate beats when one sentence would create a comma-but or broad-summary closure.",
            "Every paragraph_blueprint step must reference the finding_contracts it resolves when applicable.",
            "Normalize all findings into positive construction steps.",
            "Make paragraph_blueprint concrete enough that a writer can follow it without guessing.",
            "Each paragraph_blueprint step must name its source_basis, must_include terms, and unsafe shape to avoid.",
            "For packed lists, plan grouped relation beats, not item-by-item sentences.",
            "For author/context anchors, plan reviewable bridges instead of blocking generation.",
            "Preserve meaning and coverage, not original sentence shape.",
        ],
    }
    return "Return valid JSON only.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _planner_decision(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    decision = payload.get("planner_decision", payload)
    return decision if isinstance(decision, dict) else {}


def _merge_decision(plan: Plan, decision: dict[str, Any]) -> Plan:
    route = dict(plan.ai_safe_route)
    route["llm_planner_decision"] = {
        "status": "ok",
        "paragraph_route": decision.get("paragraph_route", ""),
        "finding_contracts": _recipe_rows(decision.get("finding_contracts")),
        "paragraph_blueprint": _recipe_rows(decision.get("paragraph_blueprint")),
        "finding_recipe_overrides": _recipe_rows(decision.get("finding_recipe_overrides")),
        "author_proxy_plan": decision.get("author_proxy_plan", ""),
        "do_not_copy_route": _string_rows(decision.get("do_not_copy_route")),
        "contract_gaps": _string_rows(decision.get("contract_gaps")),
    }
    return replace(plan, ai_safe_route=route)


def _with_planner_status(plan: Plan, status: dict[str, Any]) -> Plan:
    route = dict(plan.ai_safe_route)
    route["llm_planner_decision"] = status
    return replace(plan, ai_safe_route=route)


def _recipe_rows(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _string_rows(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(row) for row in value if str(row).strip()]


def _compact_rows(value: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in value[:limit]:
        if not isinstance(item, dict):
            continue
        rows.append({
            key: item.get(key)
            for key in (
                "id",
                "beat_id",
                "recipe_id",
                "source_sentence_id",
                "finding_tags",
                "coverage_terms",
                "starter_terms",
                "build_route",
                "positive_pattern",
            )
            if key in item
        })
    return rows


def _planner_contract_gaps(decision: dict[str, Any], findings: list[Finding]) -> list[str]:
    contracts = _recipe_rows(decision.get("finding_contracts"))
    reasons: list[str] = []
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
    return reasons[:8]


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


def _planning_label(text: str) -> str:
    labels = {"relation", "beat", "anchor", "route", "contract", "source term", "writer"}
    lowered = str(text or "").casefold()
    for label in labels:
        if label in lowered:
            return label
    return ""
