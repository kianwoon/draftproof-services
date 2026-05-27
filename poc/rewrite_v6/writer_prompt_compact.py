from __future__ import annotations

from typing import Any

from .text import Paragraph, source_anchor_terms


def source_units(paragraph: Paragraph) -> list[dict[str, Any]]:
    return [
        {
            "sentence_id": sentence.id,
            "index": sentence.index,
            "text": sentence.text,
        }
        for sentence in paragraph.sentences
    ]


def writer_execution_contract(
    paragraph: Paragraph,
    *,
    coverage_beats: list[dict[str, Any]],
    sentence_plan: list[dict[str, Any]],
    construction_recipes: list[dict[str, Any]],
    author_route_questions: list[dict[str, Any]],
    planner_decision: dict[str, Any],
) -> dict[str, Any]:
    beats_by_source = _rows_by_source(coverage_beats)
    recipes_by_source = _rows_by_source(construction_recipes)
    questions_by_source = _rows_by_source(author_route_questions)
    contracts_by_source = _rows_by_source(planner_decision.get("finding_contracts", []) if isinstance(planner_decision, dict) else [])
    blueprint_by_source = _blueprint_by_source(planner_decision.get("paragraph_blueprint", []) if isinstance(planner_decision, dict) else [])
    plan_by_source = {str(row.get("source_sentence_id") or ""): row for row in sentence_plan if isinstance(row, dict)}
    rows: list[dict[str, Any]] = []
    for sentence in paragraph.sentences:
        source_id = sentence.id
        slot = plan_by_source.get(source_id, {})
        beats = beats_by_source.get(source_id, [])
        recipes = recipes_by_source.get(source_id, [])
        questions = questions_by_source.get(source_id, [])
        contracts = contracts_by_source.get(source_id, [])
        blueprint = blueprint_by_source.get(source_id, [])
        rows.append({
            "source_sentence_id": source_id,
            "source_ref": source_id,
            "finding_tags": _dedupe([
                tag
                for row in [*beats, *recipes, *questions, *contracts]
                for tag in _strings(row.get("finding_tags") or row.get("findings"), 8)
            ])[:10],
            "coverage_beat_ids": _dedupe([
                str(beat.get("beat_id"))
                for beat in beats
                if beat.get("beat_id")
            ]),
            "coverage_terms": _source_terms_for_row(sentence.text, beats, slot),
            "must_cover_terms": _strings(slot.get("must_cover_terms"), 10),
            "route_goal": _first_text([
                slot.get("slot_goal"),
                *(row.get("safe_rebuild_shape") for row in contracts),
                *(row.get("safe_sentence_shape") for row in blueprint),
            ]),
            "sentence_route": _first_text([
                slot.get("sentence_route"),
                *(row.get("function") for row in blueprint),
            ]),
            "writer_must_do": _dedupe([
                value
                for row in contracts
                for value in _strings(row.get("writer_must_do"), 2)
            ])[:4],
            "writer_must_not_do": _dedupe([
                value
                for row in contracts
                for value in _strings(row.get("writer_must_not_do"), 2)
            ])[:4],
            "recipe_ids": _dedupe([
                str(row.get("recipe_id") or row.get("construction_recipe_id"))
                for row in [*recipes, *beats]
                if row.get("recipe_id") or row.get("construction_recipe_id")
            ]),
            "build_steps": _dedupe([
                step
                for recipe in recipes
                for step in _strings(recipe.get("build_steps"), 3)
            ])[:4],
            "route_questions": [
                {
                    "question_id": row.get("question_id"),
                    "question": row.get("question"),
                    "writer_duty": row.get("writer_duty"),
                }
                for row in questions[:2]
            ],
            "required_groups": _compact_required_groups(slot.get("required_sentence_groups")),
        })
    return {
        "source_reference_rule": "Use source_units for source text. All rows below refer to source_sentence_id; do not expect repeated source text inside findings.",
        "rows": [_drop_empty(row) for row in rows],
        "planner_gaps": _strings(planner_decision.get("contract_gaps"), 8) if isinstance(planner_decision, dict) else [],
        "do_not_copy_phrases": _strings(planner_decision.get("do_not_copy_phrases"), 8) if isinstance(planner_decision, dict) else [],
        "document_signal_contracts": planner_decision.get("document_signal_contracts", []) if isinstance(planner_decision, dict) else [],
    }


def compact_planner_decision(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    rows = [_compact_contract(row) for row in value.get("finding_contracts", []) if isinstance(row, dict)]
    blueprint = [_compact_blueprint(row) for row in value.get("paragraph_blueprint", []) if isinstance(row, dict)]
    decision = {
        "status": value.get("status"),
        "fallback_instruction": value.get("fallback_instruction"),
        "contract_gaps": _strings(value.get("contract_gaps"), 8),
        "do_not_copy_phrases": _strings(value.get("do_not_copy_phrases"), 8),
        "finding_contracts": rows[:8],
        "paragraph_blueprint": blueprint[:8],
        "document_signal_contracts": value.get("document_signal_contracts", []),
    }
    return {key: val for key, val in decision.items() if val not in (None, [], "")}


def compact_sentence_plan(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [_compact_slot(row) for row in rows if isinstance(row, dict)]


def compact_coverage_loss(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [
        {
            "source_sentence_id": row.get("source_sentence_id"),
            "group_id": row.get("group_id"),
            "coverage_beat_ids": _strings(row.get("coverage_beat_ids"), 4),
            "must_cover_terms": _strings(row.get("must_cover_terms"), 5),
            "source_terms_to_carry": _strings(row.get("source_terms_to_carry"), 12),
            "failure": row.get("failure"),
        }
        for row in rows if isinstance(row, dict)
    ]


def compact_construction_recipes(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compact.append({
            "recipe_id": row.get("recipe_id"),
            "source_sentence_id": row.get("source_sentence_id"),
            "finding_tags": _strings(row.get("finding_tags"), 6),
            "repair_sequence": [
                {
                    "repair_class": step.get("repair_class"),
                    "operator": step.get("operator"),
                }
                for step in row.get("repair_sequence", [])[:4]
                if isinstance(step, dict)
            ],
            "coverage_terms": _strings(row.get("coverage_terms"), 14),
            "build_route": row.get("build_route"),
            "build_steps": _strings(row.get("build_steps"), 3),
        })
    return compact


def _rows_by_source(rows: Any) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(rows, list):
        return by_source
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("source_sentence_id") or "")
        if source_id:
            by_source.setdefault(source_id, []).append(row)
    return by_source


def _blueprint_by_source(rows: Any) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(rows, list):
        return by_source
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_id = _source_from_blueprint(row)
        if source_id:
            by_source.setdefault(source_id, []).append(row)
    return by_source


def _source_from_blueprint(row: dict[str, Any]) -> str:
    direct = row.get("source_sentence_id")
    if direct:
        return str(direct)
    for value in _strings(row.get("source_basis"), 6):
        if "_s" in value:
            return value
    return ""


def _source_terms_for_row(sentence_text: str, beats: list[dict[str, Any]], slot: dict[str, Any]) -> list[str]:
    terms = [
        term
        for beat in beats
        for term in _strings(beat.get("coverage_terms"), 24)
    ] or _strings(slot.get("source_terms_to_carry"), 24)
    return _dedupe(terms or source_anchor_terms(sentence_text, term_limit=16))[:24]


def _compact_required_groups(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, Any]] = []
    for group in value[:8]:
        if not isinstance(group, dict):
            continue
        rows.append({
            "group_id": group.get("group_id"),
            "coverage_beat_ids": _strings(group.get("coverage_beat_ids"), 4),
            "must_cover_terms": _strings(group.get("must_cover_terms"), 8),
            "source_terms_to_carry": _strings(group.get("source_terms_to_carry"), 12),
        })
    return rows


def _first_text(values: list[Any]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _dedupe(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            rows.append(text)
            seen.add(key)
    return rows


def _drop_empty(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value not in (None, "", [], {})}


def _compact_contract(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": row.get("finding_id"),
        "source_sentence_id": row.get("source_sentence_id"),
        "finding_tags": _strings(row.get("finding_tags"), 6),
        "safe_rebuild_shape": row.get("safe_rebuild_shape"),
        "writer_must_do": _strings(row.get("writer_must_do"), 2),
        "writer_must_not_do": _strings(row.get("writer_must_not_do"), 2),
        "coverage_terms": _strings(row.get("coverage_terms"), 10),
    }


def _compact_blueprint(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step_id": row.get("step_id"),
        "function": row.get("function"),
        "route_question_id": row.get("route_question_id"),
        "source_basis": _strings(row.get("source_basis"), 4),
        "must_include": _strings(row.get("must_include"), 8),
        "must_avoid_shape": _strings(row.get("must_avoid_shape"), 2),
        "safe_sentence_shape": row.get("safe_sentence_shape"),
    }


def _compact_slot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "slot_id": row.get("slot_id"),
        "source_sentence_id": row.get("source_sentence_id"),
        "coverage_beat_ids": _strings(row.get("coverage_beat_ids"), 6),
        "slot_goal": row.get("slot_goal"),
        "sentence_route": row.get("sentence_route"),
        "source_terms_to_carry": _strings(row.get("source_terms_to_carry"), 14),
        "must_cover_terms": _strings(row.get("must_cover_terms"), 6),
        "required_sentence_groups": [
            {
                "coverage_beat_ids": _strings(group.get("coverage_beat_ids"), 4),
                "group_id": group.get("group_id"),
                "source_terms_to_carry": _strings(group.get("source_terms_to_carry"), 10),
                "must_cover_terms": _strings(group.get("must_cover_terms"), 5),
            }
            for group in row.get("required_sentence_groups", [])
            if isinstance(group, dict)
        ],
        "starter_terms": _strings(row.get("starter_terms"), 6),
    }


def _strings(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()][:limit]
