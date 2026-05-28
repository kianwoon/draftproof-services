from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Any, Protocol

from .json_io import parse_json
from .plan import Plan
from .scan import Finding
from .text import Paragraph, source_anchor_terms


class ChatClient(Protocol):
    def chat(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> Any:
        ...


def author_proxy_required(plan: Plan) -> bool:
    grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    has_pack = bool(plan.paragraph_strategy.get("author_proxy_pack"))
    return isinstance(grounding, dict) and bool(grounding.get("required")) and not has_pack


def attach_author_proxy_pack(paragraph: Paragraph, plan: Plan, findings: list[Finding], *, client: ChatClient) -> Plan:
    if not author_proxy_required(plan):
        return plan
    author_profile = _author_profile_for_plan(paragraph, plan, findings)
    response = client.chat(
        _build_prompt(paragraph, plan, findings, author_profile=author_profile),
        system=(
            "Return valid JSON only. Act as an author-proxy content layer, not the final prose writer. "
            "Act like the author and a subject expert supplying missing intent/context, not a generic essay improver. "
            "Feed the planner a small reviewable content bridge that fills the local anchor gap from submitted context."
        ),
        temperature=0.05,
        top_p=0.8,
        max_tokens=1400,
        response_format={"type": "json_object"},
        app_label="Author-proxy",
    )
    pack = _parse_pack(parse_json(getattr(response, "raw_content", "") or response.content))
    pack["author_profile"] = author_profile
    strategy = dict(plan.paragraph_strategy)
    strategy["author_proxy_pack"] = pack
    return replace(plan, paragraph_strategy=strategy)


def _build_prompt(
    paragraph: Paragraph,
    plan: Plan,
    findings: list[Finding],
    *,
    author_profile: dict[str, Any] | None = None,
) -> str:
    grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    bridge_anchors = grounding.get("bridge_anchors", []) if isinstance(grounding, dict) else []
    selected_terms = source_anchor_terms(paragraph.text, term_limit=32, phrase_limit=4)
    profile = author_profile or _author_profile(
        paragraph,
        findings,
        bridge_anchors=bridge_anchors,
        selected_terms=selected_terms,
    )
    payload = {
        "task": "author_proxy_grounding_pack",
        "role": (
            "Act like the author implied by author_profile, with only the subject expertise supported by the paragraph "
            "and nearby submitted context. Do not rewrite the paragraph. Give the planner concise content that explains "
            "what the abstract claim refers to. Prefer lived intent, practical cause, actor/object role, or source-specific "
            "reasoning over generic importance language."
        ),
        "author_profile": profile,
        "selected_paragraph": paragraph.text,
        "findings": [
            {
                "sentence_id": finding.sentence_id,
                "tags": finding.tags,
                "severity": finding.severity,
                "evidence": finding.evidence,
            }
            for finding in findings[:8]
        ],
        "available_bridge_anchors": bridge_anchors,
        "selected_paragraph_terms": selected_terms,
        "contract": {
            "do_not_write_final_paragraph": True,
            "bridge_scope": (
                "Provide a short author-expert bridge idea grounded in author_profile and submitted text. "
                "It should fill the content gap, not copy the anchor wording."
            ),
            "do_not_create": [
                "broad filler claims",
                "new thesis sentences",
                "reward-system abstractions unless present in submitted text",
                "generic conclusions",
                "generic role inflation such as essential, vital, critical filter, mentor, digital-media context, or flood of information",
                "keyword lists",
                "copied comma lists from bridge anchors",
                "instructions to omit source paragraph terms or source closing claims",
            ],
            "planner_use": (
                "The planner may use at most one bridge if it improves source flow. "
                "Group long anchor lists into a compact author bridge idea; do not pass the full list downstream. "
                "The bridge must identify which source sentence or claim it grounds, so the planner can fold the idea into that beat. "
                "Do not tell the planner to drop source paragraph terms; the bridge supplements source coverage and never replaces it. "
                "If every bridge would make the paragraph more generic or more list-heavy, return no usable_bridges and explain why. "
                "A good bridge should sound like the author explaining what they meant, not a model praising the topic."
            ),
        },
        "output_schema": {
            "proxy_pack": {
                "usable_bridges": [
                    {
                        "bridge_id": "b001",
                        "anchor_source": "selected_paragraph | previous_paragraph | next_paragraph",
                        "anchor_text": "submitted text that supports the bridge",
                        "usable_bridge": "short phrase or clause the planner may use as grounding evidence",
                        "target_sentence_ids": ["source sentence id whose claim this bridge grounds"],
                        "integration_role": "scope broad claim | ground contrast | connect cause to consequence | clarify actor/object",
                        "integration_instruction": "how the planner should use the bridge as paragraph strategy without dictating exact prose",
                        "review_reason": "why the author should review this bridge",
                    }
                ],
                "planner_guidance": {
                    "use": "one sentence of practical guidance for the planner",
                    "avoid": ["specific wording or route to avoid"],
                },
                "author_review_items": [
                    {
                        "provenance": "inferred_from_draft",
                        "target_text": "bridge phrase requiring review",
                        "user_input_needed": "confirm, replace, or remove before final submission",
                    }
                ],
            }
        },
    }
    return "Return JSON with one proxy_pack object.\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _author_profile_for_plan(paragraph: Paragraph, plan: Plan, findings: list[Finding]) -> dict[str, Any]:
    grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    bridge_anchors = grounding.get("bridge_anchors", []) if isinstance(grounding, dict) else []
    selected_terms = source_anchor_terms(paragraph.text, term_limit=32, phrase_limit=4)
    return _author_profile(paragraph, findings, bridge_anchors=bridge_anchors, selected_terms=selected_terms)


def _author_profile(
    paragraph: Paragraph,
    findings: list[Finding],
    *,
    bridge_anchors: list[Any],
    selected_terms: list[str],
) -> dict[str, Any]:
    finding_tags = _dedupe_text(
        tag
        for finding in findings
        for tag in finding.tags
        if str(tag).strip()
    )
    evidence_terms = _dedupe_text(
        term
        for finding in findings
        for term in source_anchor_terms(str(finding.evidence.get("text") or ""), term_limit=8, phrase_limit=1)
    )
    nearby_terms = _dedupe_text(
        term
        for anchor in bridge_anchors
        for term in source_anchor_terms(str(anchor), term_limit=10, phrase_limit=2)
    )
    context_terms = _dedupe_text([*selected_terms, *evidence_terms, *nearby_terms])[:28]
    actor_terms = _actor_terms(context_terms)
    return {
        "profile_basis": "selected_paragraph + paragraph_findings + nearby_submitted_anchors",
        "implied_author_position": (
            "Speak as the same author working inside this local context, not as a general essay improver."
        ),
        "local_context_terms": context_terms,
        "actor_or_subject_terms": actor_terms,
        "nearby_context_terms": nearby_terms[:12],
        "finding_focus": finding_tags[:10],
        "expertise_scope": _expertise_scope(finding_tags),
        "grounding_priorities": _grounding_priorities(finding_tags, actor_terms),
        "profile_constraints": [
            "Infer only from selected paragraph and nearby submitted anchors.",
            "Supply missing author intent or subject context as a reviewable bridge.",
            "Do not introduce a new thesis, invented evidence, or a polished generic role label.",
            "Keep the bridge short enough for the planner to fold into one source beat.",
        ],
    }


def _actor_terms(terms: list[str]) -> list[str]:
    actorish: list[str] = []
    for term in terms:
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", str(term))
        if not words:
            continue
        value = " ".join(words)
        if value.casefold().endswith(("er", "ers", "or", "ors", "ist", "ists", "ant", "ants", "ent", "ents", "ian", "ians")):
            actorish.append(term)
    return _dedupe_text(actorish)[:8]


def _expertise_scope(tags: list[str]) -> list[str]:
    scope = ["explain the local relationship between the paragraph's actors, objects, and consequence"]
    if "author_anchor_gap" in tags:
        scope.append("infer the author's missing reason from submitted context and mark it for review")
    if "context_anchor_gap" in tags:
        scope.append("name the concrete antecedent behind vague claims")
    if "broad_claim" in tags:
        scope.append("narrow broad claims to the paragraph's own context")
    if "packed_list" in tags:
        scope.append("group related items by role instead of expanding a catalogue")
    if "paragraph_rhythm" in tags or "repeated_sentence_frame" in tags:
        scope.append("support a natural paragraph route without adding decorative wording")
    return scope[:6]


def _grounding_priorities(tags: list[str], actor_terms: list[str]) -> list[str]:
    priorities = []
    if actor_terms:
        priorities.append("keep the bridge centered on local actors: " + ", ".join(actor_terms[:4]))
    if "author_anchor_gap" in tags or "context_anchor_gap" in tags:
        priorities.append("fill the anchor gap with a concrete cause, condition, or role from the submitted context")
    if "broad_claim" in tags:
        priorities.append("turn broad concern language into a scoped reason the author can confirm")
    if not priorities:
        priorities.append("add only the smallest context bridge needed for planner strategy")
    return priorities[:4]


def _dedupe_text(values: Any) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows


def _parse_pack(payload: Any) -> dict[str, Any]:
    pack = payload.get("proxy_pack") if isinstance(payload, dict) else {}
    if not isinstance(pack, dict):
        pack = {}
    bridges = [_bridge(row) for row in pack.get("usable_bridges", []) if isinstance(row, dict)]
    review_items = [_review_item(row) for row in pack.get("author_review_items", []) if isinstance(row, dict)]
    guidance = pack.get("planner_guidance") if isinstance(pack.get("planner_guidance"), dict) else pack.get("writer_guidance")
    guidance = guidance if isinstance(guidance, dict) else {}
    return {
        "usable_bridges": [row for row in bridges if row],
        "planner_guidance": {
            "use": str(guidance.get("use") or "").strip(),
            "avoid": [str(item).strip() for item in guidance.get("avoid", []) if str(item).strip()][:8]
            if isinstance(guidance.get("avoid"), list)
            else [],
        },
        "author_review_items": [row for row in review_items if row],
    }


def _bridge(row: dict[str, Any]) -> dict[str, Any]:
    usable_bridge = str(row.get("usable_bridge") or "").strip()
    anchor_text = str(row.get("anchor_text") or "").strip()
    if not usable_bridge or not anchor_text:
        return {}
    if usable_bridge.count(",") >= 3:
        return {}
    if _generic_bridge_inflation(usable_bridge):
        return {}
    unsupported_terms = _unsupported_bridge_terms(usable_bridge, anchor_text)
    if len(unsupported_terms) >= 2 and _unsupported_ratio(usable_bridge, anchor_text) >= 0.45:
        return {}
    return {
        "bridge_id": str(row.get("bridge_id") or "").strip()[:24],
        "anchor_source": str(row.get("anchor_source") or "").strip()[:40],
        "anchor_text": anchor_text[:280],
        "usable_bridge": usable_bridge[:180],
        "target_sentence_ids": _target_sentence_ids(row.get("target_sentence_ids")),
        "integration_role": str(row.get("integration_role") or "").strip()[:80],
        "integration_instruction": str(row.get("integration_instruction") or "").strip()[:220],
        "review_reason": str(row.get("review_reason") or "").strip()[:220],
    }


def _unsupported_bridge_terms(bridge: str, anchor_text: str) -> list[str]:
    anchor_terms = {_term_root(term) for term in _content_terms(anchor_text)}
    unsupported: list[str] = []
    for term in _content_terms(bridge):
        root = _term_root(term)
        if root not in anchor_terms:
            unsupported.append(term)
    return unsupported


def _unsupported_ratio(bridge: str, anchor_text: str) -> float:
    terms = _content_terms(bridge)
    if not terms:
        return 0.0
    return len(_unsupported_bridge_terms(bridge, anchor_text)) / len(terms)


def _generic_bridge_inflation(text: str) -> bool:
    lowered = str(text or "").casefold()
    generic_role_terms = {
        "critical",
        "essential",
        "vital",
        "indispensable",
        "navigator",
        "navigators",
        "mentor",
        "mentors",
        "filter",
        "filters",
        "flood",
        "underscores",
        "highlights",
        "proves",
        "evolving",
        "demands",
    }
    terms = set(_content_terms(lowered))
    if terms & generic_role_terms:
        return True
    return bool(
        re.search(r"\bdigital[-\s]+media\s+(?:context|era|environment)\b", lowered)
        or re.search(r"\bflood\s+of\s+(?:digital\s+)?(?:media|information)\b", lowered)
        or re.search(r"\b(?:teachers?|educators?)\s+(?:are|become)\s+(?:critical|essential|vital)\b", lowered)
    )


def _content_terms(text: str) -> list[str]:
    stop = {
        "about", "after", "also", "because", "before", "between", "could", "does",
        "from", "have", "into", "more", "must", "only", "over", "should", "that",
        "than", "their", "these", "they", "this", "those", "through", "with", "would",
    }
    terms = []
    for match in re.finditer(r"[A-Za-z][A-Za-z'’-]*", str(text or "")):
        value = match.group(0).casefold().strip("'’")
        if len(value) >= 4 and value not in stop:
            terms.append(value)
    return terms


def _term_root(term: str) -> str:
    value = term.casefold()
    for suffix in ("ing", "ized", "ised", "izes", "ises", "ed", "es", "s"):
        if len(value) > len(suffix) + 4 and value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _target_sentence_ids(value: Any) -> list[str]:
    if isinstance(value, str):
        rows = [value]
    elif isinstance(value, list):
        rows = value
    else:
        rows = []
    return [str(item).strip()[:40] for item in rows if str(item).strip()][:4]


def _review_item(row: dict[str, Any]) -> dict[str, str]:
    target_text = str(row.get("target_text") or row.get("generated_text") or "").strip()
    if not target_text:
        return {}
    return {
        "provenance": str(row.get("provenance") or "inferred_from_draft").strip(),
        "target_text": target_text[:180],
        "user_input_needed": str(row.get("user_input_needed") or "Confirm, replace, or remove before final submission.").strip()[:220],
    }
