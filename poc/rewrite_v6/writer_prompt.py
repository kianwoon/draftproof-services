from __future__ import annotations

import json
import os
import re
from typing import Any

from .paragraph_architecture import architecture_split_contract
from .plan import Plan
from .prompt_shape import coverage_loss_contract, paragraph_sentence_plan
from .text import Paragraph, source_terms, strip_leading_heading
from .writer_generation_rules import writer_generation_rules
from .writer_prompt_compact import (
    compact_document_signal_contracts,
    compact_planner_decision,
    source_units,
    writer_execution_contract,
)


def build_prompt(paragraph: Paragraph, plan: Plan, *, variant_focus: dict[str, str] | None = None) -> str:
    golden_route = plan.ai_safe_route.get("golden_route", {})
    coverage_beats = _prompt_coverage_beats(plan)
    sentence_plan, split_contract = paragraph_sentence_plan(paragraph, coverage_beats), architecture_split_contract(paragraph, plan)
    planner_decision = compact_planner_decision(plan.ai_safe_route.get("llm_planner_decision", {}))
    planner_decision.setdefault("repair_unit", plan.paragraph_strategy.get("repair_unit"))
    planner_decision.setdefault("flow_plan", plan.paragraph_strategy.get("dense_paragraph_plan", {}).get("flow_plan", []))
    planner_decision.setdefault("semantic_role_map", plan.paragraph_strategy.get("dense_paragraph_plan", {}).get("semantic_role_map", {}))
    planner_decision.setdefault("human_route", plan.paragraph_strategy.get("dense_paragraph_plan", {}).get("human_route", {}))
    author_proxy_grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    construction_recipes = _prompt_construction_recipes(plan)
    author_route_questions = _prompt_author_route_questions(plan)
    coverage_groups = coverage_loss_contract(sentence_plan)
    paragraph_repair_unit = plan.paragraph_strategy.get("repair_unit", "sentence_cluster")
    dense_paragraph_plan = plan.paragraph_strategy.get("dense_paragraph_plan", {})
    variant_requirements = [variant_focus] if variant_focus else _requested_variant_requirements()
    payload = {
        "task": "coverage_beat_paragraph_generation",
        "golden_question": golden_route.get(
            "question_rule",
            "What did the writer see, read, compare, struggle with, or decide, and why does it matter?",
        ),
        "golden_route": {
            "compact_formula": golden_route.get("compact_formula", "Scope it. Anchor it. Show it. Explain it. Close it."),
            "technical_formula": golden_route.get(
                "technical_formula",
                "narrow claim -> human/context anchor -> concrete evidence -> decompressed reasoning -> non-generic close",
            ),
        },
        "context_anchors": {
            "context_terms": _prompt_context_terms(plan)[:5],
            "named_references": _prompt_named_references(plan),
            "years": plan.author_proxy_context.get("years", []),
            "citation_spans": plan.author_proxy_context.get("citation_spans", []),
            "quoted_terms": plan.author_proxy_context.get("quoted_terms", []),
        },
        "content_word_boundary": {
            "allowed_content_terms": _allowed_content_terms(paragraph, plan),
            "rule": (
                "Use these terms and ordinary function words as the content boundary. "
                "Any new content noun, adjective, or abstract label must be necessary bridge wording and listed in author_review_items."
            ),
        },
        "polarity_constraints": _polarity_constraints(paragraph),
        "source_units": source_units(paragraph),
        "paragraph_repair_plan": {
            "repair_unit": paragraph_repair_unit,
            "target_sentence_range": plan.paragraph_strategy.get("target_sentence_range"),
            "dense_paragraph_plan": dense_paragraph_plan,
            "rule": (
                "When repair_unit is paragraph, scanner findings are symptoms of paragraph flow. "
                "Preserve traceability, but do not write one final sentence per finding, source row, or coverage beat."
            ),
            "semantic_role_rule": (
                "Use dense_paragraph_plan.semantic_role_map to combine source terms by role before writing. "
                "A natural compact list is allowed when the listed terms share one role; do not create catalogue sentences to avoid commas. "
                "Do not move terms from one role into another role's sentence just to satisfy coverage."
            ),
            "human_route_rule": (
                "Follow dense_paragraph_plan.human_route.movement and sentence_jobs. "
                "The route must read like lived reasoning, not topic -> list -> explanation -> neat conclusion."
            ),
            "author_proxy_grounding": author_proxy_grounding,
        },
        "writer_execution_contract": writer_execution_contract(
            paragraph,
            coverage_beats=coverage_beats,
            sentence_plan=sentence_plan,
            construction_recipes=construction_recipes,
            author_route_questions=author_route_questions,
            planner_decision=planner_decision,
        ),
        "architecture_split_contract": split_contract,
        "hard_generation_requirements": {
            "required_variant_ids": [row["id"] for row in variant_requirements],
            "exact_variant_count": len(variant_requirements),
            "list_contract_active": _list_contract_active(plan) and paragraph_repair_unit != "paragraph",
            "list_contract_rule": (
                "When paragraph_repair_plan.repair_unit is paragraph, a compact natural list is allowed if all items share one semantic role. "
                "Otherwise, no final text sentence may contain two or more commas when list_contract_active is true."
            ),
            "forbidden_sentence_starts": [
                "And",
                "But",
                "Or",
                "Which",
                "Where",
                "In",
                "Through",
                "During",
                "From",
                "This",
                "That",
                "These",
                "Those",
                "As",
                "Thereby",
            ],
            "overload_repair_rule": (
                "If one source sentence carries claim, support, event, and consequence together, split by function. "
                "Do not preserve the source sentence as a single long not-only, comma-list, or prepositional-opener sentence. "
                "For not-only/but-also source wording, keep both meaning sides through separate ordinary sentences. "
                "Do not place source work and its consequence in a pronoun-led also sentence. "
                "Do not route through repeated summary nouns such as example, result, or experience when source activities or actors are available."
            ),
            "source_preserved_shape_is_failure": "Do not return the original sentence route with synonyms. Change clause route and sentence grouping.",
            "internal_language_ban": (
                "Final text must be normal prose only. Never expose internal labels or phrases such as coverage beat, "
                "source slot, route question, relationship sees, writer_execution_contract, construction recipe, "
                "planner decision, source sentence, active variant, beat plan, or coverage capsule. Do not turn route verbs into "
                "sentence subjects, such as Guide prompts, Compare prompts, Apply knowledge occurs, or Focus expands."
            ),
            "dense_paragraph_repair_rule": (
                "If paragraph_repair_plan.repair_unit is paragraph, rebuild paragraph flow from source meaning. "
                "Merge only semantically dependent or mechanically repetitive short beats, keep independent contrasts visible, "
                "allow one natural role-based list when clearer, and avoid a chain of same-shape short sentences."
            ),
            "coverage_role_integrity_rule": (
                "Coverage must be grammatical and role-correct. A leftover source term appended to an unrelated closure, "
                "challenge, or trust sentence is a failed variant. When a term cannot fit naturally, keep prose intact and "
                "record the term in author_review_items."
            ),
        },
        "active_variant": variant_focus or {},
        "variant_requirements": variant_requirements,
        "generation_rules": writer_generation_rules(),
        "fragment_safety_contract": (
            "No final sentence may start with And, But, Or, Which, Where, That, or As. "
            "Every final sentence must stand alone with its own subject and predicate."
        ),
        "plain_route_contract_for_v1": "Low-polish source-coverage candidate: use source terms and plain relation verbs; avoid unsubmitted intensity, benefit claims, semicolon lists, and descriptor lists.",
        "author_proxy_policy": {
            "enabled": True,
            "grounding_required": bool(author_proxy_grounding.get("required")) if isinstance(author_proxy_grounding, dict) else False,
            "neighbor_bridge_anchors": author_proxy_grounding.get("bridge_anchors", []) if isinstance(author_proxy_grounding, dict) else [],
            "non_interrupting": True,
            "allowed_provenance": ["source_preserved", "inferred_from_draft", "needs_author_confirmation", "must_replace"],
            "review_required": True,
            "responsibility_boundary": "DraftProof drafts and labels provenance; the user must review and owns facts, citations, anchors, and author-proxy bridges before submission.",
            "rule": (
                "Do not stop for questions. When a needed anchor is not directly supported by submitted text, continue the rewrite and include it in author_review_items. "
                "Use author_proxy_provenance or author_review_items for reviewable bridges. "
                "If grounding_required is true, use one bridge anchor from neighbor_bridge_anchors to ground the anchor-gap paragraph; source-only synonym rotation is a failed variant."
            ),
        },
        "required_shape": {
            "paragraph_count": split_contract["paragraph_count"] if split_contract["active"] else 1,
            "sentence_count": "uncapped during golden-route discovery; use as many ordinary prose sentences as needed to preserve coverage and answer the route questions",
            "preferred_sentence_count": _preferred_sentence_count(paragraph, plan),
            "word_count": "uncapped during golden-route discovery; preserve source coverage and expand only when needed for grounding, concrete detail, or reasoning",
        },
        "output_schema": {"variants": [_variant_schema(requirement) for requirement in variant_requirements]},
    }
    if planner_decision:
        signal_contracts = planner_decision.get("document_signal_contracts", []) if isinstance(planner_decision, dict) else []
        planner_decision_payload = dict(planner_decision)
        planner_decision_payload.pop("document_signal_contracts", None)
        if planner_decision_payload:
            payload["planner_decision"] = planner_decision_payload
        if signal_contracts:
            payload["document_signal_contracts"] = compact_document_signal_contracts(signal_contracts)
    if coverage_groups:
        payload["coverage_loss_contract"] = coverage_groups
    prefix = (
        "Return valid JSON only. The JSON must contain a variants array with exactly "
        + ", ".join(row["id"] for row in variant_requirements)
        + ".\n"
    )
    return prefix + json.dumps(payload, ensure_ascii=False, indent=2)


def build_retry_contract(paragraph: Paragraph, plan: Plan) -> dict[str, Any]:
    coverage_beats = _prompt_coverage_beats(plan)
    sentence_plan = paragraph_sentence_plan(paragraph, coverage_beats)
    planner_decision = compact_planner_decision(plan.ai_safe_route.get("llm_planner_decision", {}))
    planner_decision.setdefault("repair_unit", plan.paragraph_strategy.get("repair_unit"))
    planner_decision.setdefault("flow_plan", plan.paragraph_strategy.get("dense_paragraph_plan", {}).get("flow_plan", []))
    planner_decision.setdefault("semantic_role_map", plan.paragraph_strategy.get("dense_paragraph_plan", {}).get("semantic_role_map", {}))
    planner_decision.setdefault("human_route", plan.paragraph_strategy.get("dense_paragraph_plan", {}).get("human_route", {}))
    author_proxy_grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    planner_decision.pop("document_signal_contracts", None)
    construction_recipes = _prompt_construction_recipes(plan)
    author_route_questions = _prompt_author_route_questions(plan)
    contract = {
        "context_anchors": {
            "context_terms": _prompt_context_terms(plan)[:5],
            "named_references": _prompt_named_references(plan),
            "years": plan.author_proxy_context.get("years", []),
            "citation_spans": plan.author_proxy_context.get("citation_spans", []),
            "quoted_terms": plan.author_proxy_context.get("quoted_terms", []),
        },
        "content_word_boundary": {
            "allowed_content_terms": _allowed_content_terms(paragraph, plan),
            "rule": "Stay inside these submitted terms unless bridge wording is necessary and listed in author_review_items.",
        },
        "polarity_constraints": _polarity_constraints(paragraph),
        "source_units": source_units(paragraph),
        "paragraph_repair_plan": {
            "repair_unit": plan.paragraph_strategy.get("repair_unit", "sentence_cluster"),
            "target_sentence_range": plan.paragraph_strategy.get("target_sentence_range"),
            "dense_paragraph_plan": plan.paragraph_strategy.get("dense_paragraph_plan", {}),
            "rule": (
                "When repair_unit is paragraph, retry as one paragraph flow. "
                "Do not repair each finding as a separate sentence."
            ),
            "semantic_role_rule": (
                "Use dense_paragraph_plan.semantic_role_map. A natural compact list is allowed when terms share one role."
            ),
            "author_proxy_grounding": author_proxy_grounding,
        },
        "writer_execution_contract": writer_execution_contract(
            paragraph,
            coverage_beats=coverage_beats,
            sentence_plan=sentence_plan,
            construction_recipes=construction_recipes,
            author_route_questions=author_route_questions,
            planner_decision=planner_decision,
        ),
        "architecture_split_contract": architecture_split_contract(paragraph, plan),
        "hard_generation_requirements": {
            "required_variant_ids": ["retry_v1"],
            "exact_variant_count": 1,
            "list_contract_active": _list_contract_active(plan) and plan.paragraph_strategy.get("repair_unit") != "paragraph",
            "forbidden_sentence_starts": [
                "And",
                "But",
                "Or",
                "Which",
                "Where",
                "In",
                "Through",
                "During",
                "From",
                "This",
                "That",
                "These",
                "Those",
                "As",
                "Thereby",
            ],
            "source_preserved_shape_is_failure": "Do not return the original sentence route with synonyms. Change clause route and sentence grouping.",
            "internal_language_ban": (
                "Final text must be normal prose only. Never expose internal labels or phrases such as coverage beat, "
                "source slot, route question, relationship sees, writer_execution_contract, construction recipe, "
                "planner decision, source sentence, active variant, beat plan, or coverage capsule. Do not turn route verbs into "
                "sentence subjects, such as Guide prompts, Compare prompts, Apply knowledge occurs, or Focus expands."
            ),
            "dense_paragraph_repair_rule": (
                "If paragraph_repair_plan.repair_unit is paragraph, merge semantically dependent short beats and remove repeated subject starts. "
                "Keep source meaning and avoid broad polish."
            ),
        },
        "author_proxy_policy": {
            "enabled": True,
            "grounding_required": bool(author_proxy_grounding.get("required")) if isinstance(author_proxy_grounding, dict) else False,
            "neighbor_bridge_anchors": author_proxy_grounding.get("bridge_anchors", []) if isinstance(author_proxy_grounding, dict) else [],
            "review_required": True,
            "rule": (
                "If grounding_required is true, use one bridge anchor from neighbor_bridge_anchors to ground the paragraph and list the bridge in author_review_items. "
                "Do not return source-only synonym rotation."
            ),
        },
        "active_variant": {"id": "retry_v1", "mode": "defect_feedback_retry"},
    }
    if planner_decision:
        contract["planner_decision"] = planner_decision
    coverage_groups = coverage_loss_contract(sentence_plan)
    if coverage_groups:
        contract["coverage_loss_contract"] = coverage_groups
    return contract


def _variant_requirements() -> list[dict[str, str]]:
    return [
        {
            "id": "v1",
            "mode": "coverage_beat_generation",
            "route": "plain coverage-beat route: group related source relations without trace-like one-beat sentences",
            "distinctive_obligation": "Use writer_execution_contract row order as the spine; merge grouped beats inside each source_sentence_id before moving to the next row. Change the opener route inside each affected row.",
        },
        {
            "id": "v2",
            "mode": "golden_question_generation",
            "route": "golden-question route: source basis, concrete detail, narrow interpretation, and careful close",
            "distinctive_obligation": "Answer writer_execution_contract route_questions as the spine; use the answer to change clause route while keeping readable source logic.",
        },
        {
            "id": "v3",
            "mode": "context_anchor_generation",
            "route": "context-anchor route: use available anchors first, then carry coverage beats without polished expansion",
            "distinctive_obligation": "Start each affected slot from the strongest available context/source antecedent; keep paragraph order unless the antecedent is inside the same source slot.",
        },
    ]


def _requested_variant_requirements() -> list[dict[str, str]]:
    return _variant_requirements()[:_requested_variant_count()]
def _requested_variant_count() -> int:
    raw = os.environ.get("DRAFTPROOF_V6_WRITER_VARIANTS", "3")
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(1, min(3, value))


def _variant_schema(requirement: dict[str, str]) -> dict[str, Any]:
    return {
        "id": requirement["id"],
        "mode": requirement["mode"],
        "text": "replacement text only; may contain paragraph breaks only when architecture_split_contract.active is true",
        "author_proxy_provenance": [],
        "author_review_items": [],
    }


def _preferred_sentence_count(paragraph: Paragraph, plan: Plan | None = None) -> str:
    if plan is not None:
        target = plan.paragraph_strategy.get("target_sentence_range")
        if target:
            return str(target)
    source_count = max(1, len(paragraph.sentences))
    low = max(1, source_count)
    high = min(source_count + 2, max(source_count, int(source_count * 1.35) + 1))
    return f"{low} to {max(low, high)} ordinary sentences"


def _prompt_coverage_beats(plan: Plan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    recent_concrete_terms: list[str] = []
    for beat in plan.ai_safe_route.get("coverage_beats", []):
        if not isinstance(beat, dict):
            continue
        terms = [str(term) for term in beat.get("coverage_terms", []) if str(term).strip()]
        finding_tags = [str(tag) for tag in beat.get("finding_tags", []) if str(tag).strip()]
        capsule_terms = _resolved_capsule_terms(terms, recent_concrete_terms)
        polarity_markers = [str(marker) for marker in beat.get("polarity_markers", []) if str(marker).strip()]
        capsule_terms = _capsule_with_polarity(capsule_terms, polarity_markers)
        intent = _coverage_intent(terms, recent_concrete_terms)
        rows.append({
            "beat_id": beat.get("beat_id"),
            "source_sentence_id": beat.get("source_sentence_id"),
            "finding_tags": finding_tags,
            "construction_recipe_id": beat.get("construction_recipe_id"),
            "construction_recipe": _compact_recipe(beat.get("construction_recipe", {})),
            "coverage_capsule": " | ".join(capsule_terms),
            "coverage_terms": terms,
            "polarity_markers": polarity_markers,
            "polarity_instruction": _polarity_instruction(polarity_markers),
            "starter_terms": beat.get("starter_terms", []),
            "grouped_relation": bool(beat.get("grouped_relation")),
            "coverage_intent": intent,
            "finding_instruction": _finding_instruction(finding_tags),
        })
        recent_concrete_terms = _remember_concrete_terms(recent_concrete_terms, terms)
    return rows


def _list_contract_active(plan: Plan) -> bool:
    return any(
        "packed_list" in [str(tag) for tag in beat.get("finding_tags", [])]
        for beat in plan.ai_safe_route.get("coverage_beats", [])
        if isinstance(beat, dict)
    )


def _compact_recipe(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "build_route": value.get("build_route"),
        "build_steps": [str(step) for step in value.get("build_steps", [])[:3]],
    }


def _polarity_constraints(paragraph: Paragraph) -> list[dict[str, Any]]:
    text = str(paragraph.text or "")
    lowered = text.casefold()
    rows: list[dict[str, Any]] = []
    if "not less" in lowered:
        rows.append({
            "source_marker": "not less",
            "required_direction": "preserve that the role, value, or condition is not reduced",
            "required_shapes": ["not less", "remains important", "still important", "more important"],
            "forbidden_shapes": ["reduced", "reducing", "limited", "limits", "limiting", "diminished", "less than"],
        })
    if "no longer" in lowered:
        rows.append({
            "source_marker": "no longer",
            "required_direction": "preserve that the old condition does not fully apply now",
            "required_shapes": ["no longer", "does not", "not fully", "not the same"],
            "forbidden_shapes": ["still fully", "continues to fully"],
        })
    if "not only" in lowered:
        rows.append({
            "source_marker": "not only",
            "required_direction": "preserve that the first side is insufficient by itself and the second side also matters",
            "required_shapes": ["both sides matter", "also", "the first side remains present plus the second side matters"],
            "forbidden_shapes": ["only", "rather than", "not enough by itself", "rewards people who remember facts, not only", "not only a concern", "not only a serious concern"],
        })
    if "rather than" in lowered or "instead of" in lowered:
        rows.append({
            "source_marker": "rather than",
            "required_direction": "preserve which side the source treats as the pressure or preference",
            "required_shapes": ["rather than", "instead of", "over"],
            "forbidden_shapes": ["reversed contrast direction"],
        })
    if "not always" in lowered:
        rows.append({
            "source_marker": "not always",
            "required_direction": "preserve the limited side of the contrast without moving the limitation onto the positive side",
            "required_shapes": ["not always", "does not always", "do not always"],
            "forbidden_shapes": ["not always learn to pass", "not always pass"],
        })
    return rows


def _allowed_content_terms(paragraph: Paragraph, plan: Plan) -> list[str]:
    terms: list[str] = []
    terms.extend(source_terms(strip_leading_heading(paragraph.text), limit=80))
    for value in _prompt_context_terms(plan)[:5]:
        terms.append(str(value))
    grounding = plan.paragraph_strategy.get("author_proxy_grounding", {})
    if isinstance(grounding, dict) and grounding.get("required"):
        for anchor in grounding.get("bridge_anchors", [])[:4]:
            if isinstance(anchor, dict):
                terms.extend(str(term) for term in anchor.get("bridge_terms", []) if str(term).strip())
    for value in _prompt_named_references(plan)[:16]:
        terms.extend(source_terms(str(value), limit=8) or [str(value)])
    for key in ("years", "citation_spans", "quoted_terms"):
        for value in plan.author_proxy_context.get(key, [])[:16]:
            terms.extend(source_terms(str(value), limit=8) or [str(value)])
    for beat in plan.ai_safe_route.get("coverage_beats", []):
        if isinstance(beat, dict):
            terms.extend(str(term) for term in beat.get("coverage_terms", []) if str(term).strip())
    return _dedupe_terms(terms)[:120]


def _dedupe_terms(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        rows.append(text)
        seen.add(key)
    return rows


def _prompt_construction_recipes(plan: Plan) -> list[dict[str, Any]]:
    rows = plan.ai_safe_route.get("construction_recipes", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _prompt_author_route_questions(plan: Plan) -> list[dict[str, Any]]:
    rows = plan.ai_safe_route.get("author_route_questions", [])
    if not isinstance(rows, list):
        return []
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compact.append({
            "question_id": row.get("question_id"),
            "source_sentence_id": row.get("source_sentence_id"),
            "finding_tags": row.get("finding_tags", []),
            "question": row.get("question"),
            "answer_basis_terms": row.get("answer_basis_terms", []),
            "writer_duty": row.get("writer_duty"),
            "author_proxy_policy": row.get("author_proxy_policy"),
        })
    return compact


def _finding_instruction(tags: list[str]) -> str:
    tag_set = set(tags)
    instructions: list[str] = []
    if "packed_list" in tag_set:
        instructions.append("Keep grouped anchors grouped; do not recombine them into a comma list and do not split into repeated one-item sentences.")
    if "context_anchor_gap" in tag_set:
        instructions.append("Name the concrete antecedent; do not start from this, that, it, they, these, or those.")
    if tag_set & {"author_anchor_gap", "unsupported_claim_gap", "broad_claim"}:
        instructions.append("Put the concrete source relation before any evaluative word; avoid broad labels like important, challenge, or concern unless a submitted anchor appears in the same sentence.")
        instructions.append("When the source beat uses an evaluative label, convert it into an observable role, pressure, decision, support, or contrast relation instead of repeating the label as the main claim.")
    if "paraphrase_smoothing" in tag_set:
        instructions.append("Use shorter source-level pressure; do not turn the beat into smoother academic explanation.")
    if "sentence_overload" in tag_set:
        instructions.append("Split at source relationships; do not summarize the beat into one longer polished sentence.")
    return " ".join(instructions)


def _capsule_with_polarity(capsule_terms: list[str], polarity_markers: list[str]) -> list[str]:
    if not polarity_markers:
        return capsule_terms
    if any(str(marker).casefold() == "not always" for marker in polarity_markers):
        return [*capsule_terms, "not always limit"]
    rows: list[str] = []
    seen: set[str] = set()
    marker_words = {
        word.casefold()
        for marker in polarity_markers
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", marker)
    }
    for value in [*polarity_markers, *capsule_terms]:
        key = value.casefold()
        if key in seen or key in marker_words and value not in polarity_markers:
            continue
        rows.append(value)
        seen.add(key)
    return rows


def _polarity_instruction(polarity_markers: list[str]) -> str:
    lowered = " ".join(polarity_markers).casefold()
    if "not only" in lowered:
        return "Keep the not-only direction with a plain not-only/also relation. Do not convert it into not enough by itself unless the source uses those exact words."
    if "not always" in lowered:
        return "Keep the limitation on the limited side only. Use one contrast sentence, not repeated not-always sentences."
    if re.search(r"\bmore\s+\w+\s+than\b", lowered):
        return "Keep the more-than comparison direction; the first side is stronger and the than-side is limited, not endorsed."
    if "rather than" in lowered or "instead of" in lowered:
        return "Keep the contrast direction; do not reverse which side is preferred or criticized."
    if "no longer" in lowered:
        return "Keep the time contrast; do not say the old condition still fully applies."
    if "without" in lowered or "not" in lowered:
        return "Keep the negation; do not turn the marked idea into an affirmative claim."
    return ""


def _resolved_capsule_terms(terms: list[str], recent_concrete_terms: list[str]) -> list[str]:
    lowered = {term.casefold() for term in terms}
    if lowered and lowered <= {"model", "exists"} and recent_concrete_terms:
        return [*recent_concrete_terms[-4:], "still exists"]
    if {"fully", "reflects"} & lowered:
        anchor = recent_concrete_terms[-3:] if recent_concrete_terms else ["previous route"]
        return [*anchor, "does not fully fit", "young people learn"]
    return terms


def _coverage_intent(terms: list[str], recent_concrete_terms: list[str]) -> str:
    lowered = {term.casefold() for term in terms}
    if lowered and lowered <= {"model", "exists"}:
        return "continuity beat: name the previous concrete route instead of writing a vague model sentence"
    if {"fully", "reflects"} & lowered:
        return "contrast beat: explain that the previous concrete route does not fully fit how young people learn now"
    return "source coverage beat"


def _remember_concrete_terms(existing: list[str], terms: list[str]) -> list[str]:
    vague = {"model", "exists", "longer", "fully", "reflects"}
    rows = [*existing]
    seen = {term.casefold() for term in rows}
    for term in terms:
        key = term.casefold()
        if key in vague or key in seen:
            continue
        rows.append(term)
        seen.add(key)
    return rows[-12:]


def _prompt_context_terms(plan: Plan) -> list[str]:
    coverage_terms = {
        str(term).casefold()
        for beat in plan.ai_safe_route.get("coverage_beats", [])
        if isinstance(beat, dict)
        for term in beat.get("coverage_terms", [])
    }
    selected_terms = {str(term).casefold() for term in plan.author_proxy_context.get("source_terms", [])}
    neighbor_named_terms = {
        term.casefold()
        for value in plan.author_proxy_context.get("named_references", [])
        for term in source_terms(str(value), limit=8)
        if term.casefold() not in selected_terms
    }
    neighbor_list_terms = _neighbor_list_terms(plan) - selected_terms
    forbidden = {
        "today", "now", "overall", "therefore", "however", "this", "that",
        "they", "these", "those", "past", "model",
    }
    rows: list[str] = []
    for term in plan.author_proxy_context.get("context_terms", []):
        key = str(term).casefold()
        if key in forbidden or key in coverage_terms:
            continue
        if key in neighbor_named_terms or key in neighbor_list_terms:
            continue
        rows.append(term)
    return rows


def _neighbor_list_terms(plan: Plan) -> set[str]:
    context = plan.author_proxy_context
    terms: set[str] = set()
    neighbor_text = "\n".join([
        str(context.get("before_context") or ""),
        str(context.get("after_context") or ""),
    ])
    for sentence in re.split(r"(?<=[.!?])\s+", neighbor_text):
        if sentence.count(",") < 2 and len(re.findall(r"\b(?:and|or|also|but)\b", sentence, flags=re.I)) < 2:
            continue
        for term in source_terms(sentence, limit=32):
            terms.add(term.casefold())
    return terms


def _prompt_named_references(plan: Plan) -> list[str]:
    selected_terms = {str(term).casefold() for term in plan.author_proxy_context.get("source_terms", [])}
    rows: list[str] = []
    for value in plan.author_proxy_context.get("named_references", []):
        terms = source_terms(str(value), limit=8)
        if any(term.casefold() in selected_terms for term in terms) or not _reference_in_neighbor_list(plan, str(value)):
            rows.append(str(value))
    return rows


def _reference_in_neighbor_list(plan: Plan, value: str) -> bool:
    needle = str(value or "").strip()
    if not needle:
        return False
    context = plan.author_proxy_context
    neighbor_text = "\n".join([
        str(context.get("before_context") or ""),
        str(context.get("after_context") or ""),
    ])
    for sentence in re.split(r"(?<=[.!?])\s+", neighbor_text):
        if needle not in sentence:
            continue
        if re.search(r"\b(?:19|20)\d{2}[a-z]?\b", sentence):
            return False
        named_count = len(re.findall(r"\b(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9'’-]{2,}|[A-Z]{2,}))*\b", sentence))
        if named_count >= 2:
            return True
        if sentence.count(",") >= 2 or len(re.findall(r"\b(?:and|or)\b", sentence, flags=re.I)) >= 2:
            return True
    return False
