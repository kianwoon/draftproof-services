from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Protocol

from .json_io import parse_json
from .plan import Plan
from .scan import Finding
from .text import Paragraph
from .writer_prompt_compact import compact_document_signal_contracts


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
            app_label="planner",
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
            "author_route_questions": _compact_rows(plan.ai_safe_route.get("author_route_questions", []), limit=16),
            "golden_route": plan.ai_safe_route.get("golden_route", {}),
            "document_signal_contracts": compact_document_signal_contracts(plan.ai_safe_route.get("document_signal_contracts", [])),
        },
        "required_decision": {
            "repair_unit": "paragraph when dense findings should be repaired as one paragraph flow; otherwise sentence_cluster",
            "paragraph_problem": "plain-language diagnosis of why sentence-level repair would fail",
            "flow_plan": [
                {
                    "step_id": "fp001",
                    "function": "paragraph-level job for this beat",
                    "source_basis": ["submitted sentence id or source term"],
                    "must_include": ["source-supported term or relation"],
                    "must_not_become": ["mechanical or AI-shaped failure to avoid"],
                }
            ],
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
                    "route_question_id": "author_route_question id this step answers, when applicable",
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
            "Return one finding_contract for every scanner_findings row for traceability.",
            "When deterministic_route_skeleton.paragraph_strategy.repair_unit is paragraph, treat finding_contracts as diagnostic symptoms feeding one paragraph flow, not as separate final sentences.",
            "For dense paragraph repair, paragraph_route and flow_plan must tell the writer how to group, merge, and sequence source ideas at paragraph level.",
            "For dense paragraph repair, do not plan one output sentence per scanner finding, one output sentence per coverage beat, or one output sentence per source item.",
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
            "Use deterministic_route_skeleton.author_route_questions as the frame for paragraph_blueprint.",
            "Every blueprint step for a scanner finding must answer a route question, not merely restate an avoid rule.",
            "When a finding needs author/context support, plan an Author-Proxy bridge from submitted anchors or mark it as reviewable.",
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
    contract_gaps = _string_rows(decision.get("contract_gaps"))
    unsafe_contracts = _unsafe_contract_gaps(contract_gaps)
    fallback_contracts = _fallback_finding_contracts(plan, contract_gaps) if unsafe_contracts else []
    fallback_blueprint = _fallback_paragraph_blueprint(plan, contract_gaps) if unsafe_contracts else []
    route["llm_planner_decision"] = {
        "status": "degraded_contract_gaps" if unsafe_contracts else "ok",
        "repair_unit": decision.get("repair_unit") or plan.paragraph_strategy.get("repair_unit"),
        "paragraph_problem": decision.get("paragraph_problem", ""),
        "flow_plan": [] if unsafe_contracts else _recipe_rows(decision.get("flow_plan")),
        "paragraph_route": decision.get("paragraph_route", ""),
        "finding_contracts": fallback_contracts if unsafe_contracts else _recipe_rows(decision.get("finding_contracts")),
        "paragraph_blueprint": fallback_blueprint if unsafe_contracts else _recipe_rows(decision.get("paragraph_blueprint")),
        "finding_recipe_overrides": [] if unsafe_contracts else _recipe_rows(decision.get("finding_recipe_overrides")),
        "document_signal_contracts": _recipe_rows(route.get("document_signal_contracts")),
        "author_proxy_plan": decision.get("author_proxy_plan", ""),
        "do_not_copy_route": _string_rows(decision.get("do_not_copy_route")),
        "do_not_copy_phrases": _do_not_copy_phrases(contract_gaps),
        "contract_gaps": contract_gaps,
        "fallback_instruction": (
            "Ignore unsafe LLM planner shapes and use the deterministic fallback finding_contracts, "
            "paragraph_blueprint, author_route_questions, coverage_beats, and construction_recipes."
        ) if unsafe_contracts else "",
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


def _unsafe_contract_gaps(gaps: list[str]) -> bool:
    unsafe_markers = (
        "comma-list route",
        "copies risky source phrase",
        "contains planning label",
        "forbidden sentence opener",
        "placeholder brackets",
        "semicolon punctuation",
        "unsubmitted bridge term",
    )
    return any(any(marker in gap for marker in unsafe_markers) for gap in gaps)


def _do_not_copy_phrases(gaps: list[str]) -> list[str]:
    phrases: list[str] = []
    for gap in gaps:
        phrases.extend(match.group(1).strip() for match in re.finditer(r"'([^']{4,160})'", gap))
    return _dedupe_strings(phrases)[:12]


def _fallback_finding_contracts(plan: Plan, gaps: list[str]) -> list[dict[str, Any]]:
    copied = _do_not_copy_phrases(gaps)
    contracts: list[dict[str, Any]] = []
    for action in plan.actions:
        if not action.tags:
            continue
        contracts.append({
            "finding_id": f"{action.sentence_id}:{','.join(action.tags)}",
            "source_sentence_id": action.sentence_id,
            "finding_tags": list(action.tags),
            "unsafe_original_shape": action.source_text,
            "safe_rebuild_shape": _fallback_safe_shape(action),
            "writer_must_do": [
                action.method,
                "preserve exact anchors and numeric/source-code terms",
                "split overload into adjacent sentence rows",
                "revoice polished or evaluative source wording as plain source meaning",
            ],
            "writer_must_not_do": _dedupe_strings([
                *action.do_not,
                "copy the original sentence route",
                "copy risky source phrase fragments",
                *copied,
            ]),
            "coverage_terms": list(action.preserve_terms),
        })
    return contracts


def _fallback_paragraph_blueprint(plan: Plan, gaps: list[str]) -> list[dict[str, Any]]:
    copied = _do_not_copy_phrases(gaps)
    steps: list[dict[str, Any]] = []
    for index, action in enumerate(plan.actions, start=1):
        if not action.tags:
            continue
        steps.append({
            "step_id": f"fallback_b{index:03d}",
            "function": action.operation,
            "route_question_id": f"{action.sentence_id}_q01",
            "source_basis": [action.sentence_id],
            "must_include": list(action.preserve_terms),
            "must_avoid_shape": _dedupe_strings([*action.do_not, *copied]),
            "safe_sentence_shape": _fallback_safe_shape(action),
        })
    return steps


def _fallback_safe_shape(action: Any) -> str:
    tags = set(getattr(action, "tags", []) or [])
    terms = list(getattr(action, "preserve_terms", []) or [])
    first, second = _shape_term_pairs(terms)
    if "citation_anchor" in tags:
        return f"Keep the submitted citation parenthetical near the supported claim using {first}. Carry the next relation with {second} without adding citation report verbs."
    if "packed_list" in tags:
        return f"Group the first relation around {first}. Carry the next relation separately with {second}."
    if "sentence_overload" in tags:
        return f"Use one sentence for the first source relation around {first}. Use the next sentence for the next relation around {second}."
    if "context_anchor_gap" in tags or "predictable_start" in tags:
        return f"Start with the concrete source anchor {first}, then state the scoped relation using {second}."
    return f"Write a direct source relation using {first}; keep any next relation separate with {second}."


def _term_pair(terms: list[str]) -> str:
    rows = [str(term).strip() for term in terms if str(term).strip()]
    if not rows:
        return "the submitted anchor"
    if len(rows) == 1:
        return rows[0]
    return " and ".join(rows[:2])


def _shape_term_pairs(terms: list[str]) -> tuple[str, str]:
    rows = _dedupe_strings([str(term).strip() for term in terms if str(term).strip()])
    if not rows:
        return "the submitted anchor", "the next submitted detail"
    weak = {
        "actually", "another", "carry", "complete", "creating", "demonstrates",
        "example", "found", "helping", "made", "shows", "thereby", "them", "view",
    }
    phrase_primary = next((term for term in rows if " " in term), "")
    primary = phrase_primary or next((term for term in rows if term.casefold() not in weak), rows[0])
    unused = [term for term in rows if term != primary]
    strong = [
        term for term in unused
        if any(ch.isdigit() for ch in term) or term.isupper() or " " in term
    ]
    details = [term for term in unused if term not in strong and term.casefold() not in weak and len(term) >= 5]
    pool = [*strong, *details, *unused] if phrase_primary else [*details, *strong, *unused]
    first_detail = (pool or ["the submitted detail"])[0]
    used = {primary, first_detail}
    tail = rows[rows.index(first_detail) + 1:] if first_detail in rows else []
    second_rows = [
        term for term in [*tail, *strong, *details, *unused]
        if term not in used
    ]
    return _term_pair([primary, first_detail]), _term_pair(second_rows[:2])


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
                "question_id",
                "question",
                "answer_basis_terms",
                "writer_duty",
                "signal_group",
                "score",
                "writer_obligation",
                "author_proxy_policy",
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
