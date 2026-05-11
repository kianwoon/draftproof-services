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

_GENERIC_EXPANSION_MARKERS = (
    "one of the",
    "another important",
    "important feature",
    "plays a",
    "major role",
    "significant",
    "influential",
    "has become",
    "is known for",
    "is also",
    "this shows",
    "this reflects",
    "in conclusion",
    "overall",
    "despite",
    "however",
    "moreover",
    "furthermore",
    "additionally",
)
_TEMPLATE_MARKERS = (
    "in conclusion",
    "overall",
    "another important",
    "one of the",
    "despite its",
    "at the same time",
    "on the other hand",
    "this is why",
)
_CANONICAL_FACT_MARKERS = (
    "was founded",
    "declared independence",
    "constitution",
    "civil war",
    "world war",
    "president",
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


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", str(text or "")))


def _sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if s.strip()])


def split_candidate_blocks(source_text: str) -> list[dict[str, Any]]:
    """Split source text into paragraph-like blocks with stable indexes.

    The returned ``part_index`` lets the assembler replace only targeted blocks
    while preserving all separators and untouched blocks from the source text.
    """

    parts = re.split(r"(\n\s*\n)", str(source_text or ""))
    blocks: list[dict[str, Any]] = []
    block_index = 0
    for part_index, text in enumerate(parts):
        if part_index % 2:
            continue
        if not str(text or "").strip():
            continue
        lower = text.lower()
        entities = named_entity_inventory(text, limit=20)
        numbers = re.findall(r"\b\d{2,4}\b", text)
        generic_hits = sum(1 for marker in _GENERIC_EXPANSION_MARKERS if marker in lower)
        template_hits = sum(1 for marker in _TEMPLATE_MARKERS if marker in lower)
        canonical_hits = sum(1 for marker in _CANONICAL_FACT_MARKERS if marker in lower)
        sentence_total = _sentence_count(text)
        words = _word_count(text)
        protected_anchor_score = min(6.0, len(entities) * 1.4 + len(numbers) * 1.2 + canonical_hits * 2.0)
        generic_density = generic_hits / max(1, sentence_total)
        template_density = template_hits / max(1, sentence_total)
        length_drag = min(3.0, max(0.0, (words - 65) / 35.0))
        weighted_drag = (
            generic_hits * 2.0
            + template_hits * 2.5
            + length_drag
            + (1.5 if sentence_total >= 4 else 0.0)
            - min(2.0, protected_anchor_score * 0.25)
        )
        if canonical_hits or len(numbers) >= 1 or len(entities) >= 3:
            role = "canonical_fact_preserve" if generic_hits <= 1 and template_hits == 0 else "fact_preserve_reframe"
        elif template_hits:
            role = "template_geometry_target"
        elif generic_hits or words > 75:
            role = "generic_expansion_target"
        else:
            role = "preserve"
        remove_safe = (
            role in {"generic_expansion_target", "template_geometry_target"}
            and protected_anchor_score < 2.0
            and words <= 90
        )
        blocks.append({
            "index": block_index,
            "part_index": part_index,
            "text": text.strip(),
            "word_count": words,
            "sentence_count": sentence_total,
            "role": role,
            "generic_hits": generic_hits,
            "template_hits": template_hits,
            "canonical_hits": canonical_hits,
            "generic_density": round(generic_density, 3),
            "template_density": round(template_density, 3),
            "protected_anchor_terms": entities[:12] + numbers[:8],
            "protected_anchor_score": round(protected_anchor_score, 3),
            "weighted_drag": round(max(0.0, weighted_drag), 3),
            "remove_safe": remove_safe,
        })
        block_index += 1
    return blocks


def block_portfolio_tasks(source_text: str, report_dict: dict | None, *, limit: int = DEFAULT_LLM_CANDIDATES) -> list[dict[str, Any]]:
    """Build five block-scoped LLM tasks from the current source/report.

    This is intentionally not content-specific: it ranks blocks by generic
    expansion, template rhythm, anchor protection, and current formula drivers.
    """

    blocks = split_candidate_blocks(source_text)
    if not blocks:
        return []
    plan = formula_gap_plan(report_dict)
    dominant = [d for d in plan.get("dominant_drivers") or [] if isinstance(d, str)]
    ranked = sorted(blocks, key=lambda b: (float(b.get("weighted_drag") or 0.0), b.get("word_count") or 0), reverse=True)
    generic_ranked = [b for b in ranked if b.get("role") in {"generic_expansion_target", "fact_preserve_reframe"}]
    template_ranked = [b for b in ranked if b.get("role") == "template_geometry_target" or b.get("template_hits")]
    removable_ranked = [b for b in ranked if b.get("remove_safe")]
    anchor_ranked = [b for b in ranked if b.get("protected_anchor_score", 0) < 4.5 and b.get("word_count", 0) >= 35]

    def pick(rows: list[dict[str, Any]], count: int = 1, *, exclude: set[int] | None = None) -> list[dict[str, Any]]:
        excluded = exclude or set()
        selected: list[dict[str, Any]] = []
        for row in rows or ranked:
            idx = int(row.get("index") or 0)
            if idx in excluded:
                continue
            selected.append(row)
            if len(selected) >= count:
                break
        return selected

    used: set[int] = set()
    specs = [
        ("STATISTICAL_TEXTURE_REBUILD", "replace", generic_ranked or ranked, 1),
        ("SEMANTIC_VARIANCE_RESTRUCTURE", "replace", template_ranked or ranked, 1),
        ("HUMAN_ANCHOR_SUPPRESSION_GAIN", "replace", anchor_ranked or generic_ranked or ranked, 1),
        ("HYBRID_TEXTURE_ANCHOR", "replace", ranked, 2),
        ("LOW_VALUE_COMPRESS_REMOVE", "compress_or_remove", removable_ranked or generic_ranked or ranked, 1),
    ]
    tasks: list[dict[str, Any]] = []
    for family, operation, rows, count in specs[: max(0, int(limit))]:
        selected = pick(rows, count, exclude=used if count == 1 else set())
        if not selected:
            selected = pick(rows or ranked, count, exclude=set())
        if not selected:
            continue
        if count == 1:
            used.update(int(row["index"]) for row in selected)
        task_blocks = [
            {
                "index": row["index"],
                "role": row["role"],
                "word_count": row["word_count"],
                "weighted_drag": row["weighted_drag"],
                "protected_anchor_terms": row["protected_anchor_terms"],
                "remove_safe": row["remove_safe"],
                "text": row["text"],
            }
            for row in selected
        ]
        target_drivers = list(dominant[:3])
        if family == "HUMAN_ANCHOR_SUPPRESSION_GAIN":
            target_drivers.append("human_anchor_suppression")
        if family == "LOW_VALUE_COMPRESS_REMOVE":
            target_drivers.extend(["patchwork_expansion", "semantic_uniformity"])
        tasks.append({
            "family": family,
            "operation": operation,
            "block_indexes": [row["index"] for row in selected],
            "targeted_drivers": list(dict.fromkeys(target_drivers)),
            "blocks": task_blocks,
            "instruction": _block_task_instruction(family, operation),
        })
    return tasks


def _block_task_instruction(family: str, operation: str) -> str:
    if family == "STATISTICAL_TEXTURE_REBUILD":
        return "Patch only the selected block to reduce predictable cadence, route symmetry, and over-smooth explanation."
    if family == "SEMANTIC_VARIANCE_RESTRUCTURE":
        return "Patch only the selected block to break repeated paragraph role and claim-explain-conclude rhythm."
    if family == "HUMAN_ANCHOR_SUPPRESSION_GAIN":
        return "Patch only the selected block with bounded implied process reasoning, limitation, or judgement from the submitted context."
    if family == "HYBRID_TEXTURE_ANCHOR":
        return "Patch only the selected blocks with a combined texture and bounded anchor-suppression move."
    if operation == "compress_or_remove":
        return "Compress the selected low-value block sharply, or remove it only if no unique fact, anchor, citation, or required claim is lost."
    return "Patch only the selected block and preserve all other blocks exactly."


def formula_gap_candidate_prompt(
    source_text: str,
    report_dict: dict | None,
    family: str,
    *,
    protected_anchors: list[dict] | None = None,
    block_task: dict[str, Any] | None = None,
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
        "patches": [
            {
                "block_index": 0,
                "operation": "replace",
                "replacement_text": "replacement for this block only",
            }
        ],
        "candidate_text": "",
    }
    block_scope_text = ""
    if isinstance(block_task, dict) and block_task.get("blocks"):
        scope_rows = []
        for row in block_task.get("blocks") or []:
            scope_rows.append({
                "block_index": row.get("index"),
                "role": row.get("role"),
                "word_count": row.get("word_count"),
                "weighted_drag": row.get("weighted_drag"),
                "protected_anchor_terms": row.get("protected_anchor_terms"),
                "remove_safe": row.get("remove_safe"),
                "text": row.get("text"),
            })
        block_scope_text = (
            "\nSelected block patch task:\n"
            f"{json.dumps({k: v for k, v in block_task.items() if k != 'blocks'}, ensure_ascii=False)[:2200]}\n"
            "Selected source blocks. Return patches only for these block_index values:\n"
            f"{json.dumps(scope_rows, ensure_ascii=False, indent=2)[:6000]}\n"
        )
    return (
        "DraftProof FORMULA_GAP_PORTFOLIO_CANDIDATE.\n"
        "Objective: produce one block-scoped candidate that lowers the shared Turnitin-like AI score below 20 if possible.\n"
        "Selection is based on a full rescan. Do not optimize a single raw signal if total weighted score gets worse.\n\n"
        f"Portfolio family: {family}\n"
        f"Family instruction: {family_instructions}\n\n"
        f"Current formula plan:\n{json.dumps(plan, ensure_ascii=False)[:3600]}\n\n"
        "Hard constraints:\n"
        "- Preserve all named entities, dates, numbers, citations, quotes, protected anchors, and core factual claims.\n"
        "- Every protected anchor listed below must remain visible in the candidate unless it is explicitly duplicated elsewhere in equivalent form.\n"
        "- Return JSON patches for the selected blocks only. Do not rewrite untouched blocks.\n"
        "- The assembler will copy unchanged blocks exactly from the source document.\n"
        "- Prefer one high-impact block patch over many small edits across the whole essay.\n"
        "- If a fact/example is hard to improve safely, keep the original wording for that fact/example.\n"
        "- Do not add fake people, fake dates, fake sources, unsupported evidence, or fabricated lived experience.\n"
        "- Do not use personal voice as a default operation.\n"
        "- You may change structure, paragraph order, pacing, and explanation density if facts remain intact.\n"
        "- Avoid generic polished transitions and balanced essay rhythm.\n"
        "- Keep argument continuity and readable academic tone.\n\n"
        f"Protected anchors:\n{json.dumps(protected_anchors or [], ensure_ascii=False)[:2600]}\n\n"
        f"{block_scope_text}\n"
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
    patches = payload.get("patches")
    has_valid_patch = False
    if isinstance(patches, list):
        for patch in patches:
            if not isinstance(patch, dict):
                continue
            if patch.get("block_index") is None:
                continue
            if str(patch.get("replacement_text") or "").strip() or str(patch.get("operation") or "").lower() == "remove":
                has_valid_patch = True
                break
    if not candidate_text and not has_valid_patch:
        return None, "missing_candidate_text_or_patches"
    if payload.get("unsupported_new_facts") is True:
        return None, "unsupported_new_facts_declared"
    for key in ("fact_inventory_preserved", "core_claims_preserved_or_merged", "protected_anchors_preserved"):
        if payload.get(key) is False:
            return None, f"{key}_false"
    return payload, ""


def assemble_candidate_from_payload(source_text: str, payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]], str]:
    """Assemble a full candidate from JSON patches while preserving untouched blocks."""

    source = str(source_text or "")
    patches = payload.get("patches") if isinstance(payload, dict) else None
    if not isinstance(patches, list) or not patches:
        candidate_text = str((payload or {}).get("candidate_text") or "").strip()
        if candidate_text and candidate_text.strip() != source.strip():
            return candidate_text, [], ""
        return "", [], "empty_or_unchanged_candidate"
    parts = re.split(r"(\n\s*\n)", source)
    blocks = split_candidate_blocks(source)
    block_part_index = {int(block["index"]): int(block["part_index"]) for block in blocks}
    applied: list[dict[str, Any]] = []
    touched: set[int] = set()
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        try:
            block_index = int(patch.get("block_index"))
        except Exception:
            continue
        if block_index not in block_part_index or block_index in touched:
            continue
        operation = str(patch.get("operation") or "replace").strip().lower()
        replacement = str(patch.get("replacement_text") or "").strip()
        original = parts[block_part_index[block_index]]
        if operation == "remove":
            replacement = ""
        if not replacement and operation != "remove":
            continue
        if replacement.strip() == original.strip():
            continue
        parts[block_part_index[block_index]] = replacement
        touched.add(block_index)
        applied.append({
            "block_index": block_index,
            "operation": operation,
            "original_word_count": _word_count(original),
            "replacement_word_count": _word_count(replacement),
        })
    candidate = "".join(parts).strip()
    if not applied:
        return "", [], "no_applicable_block_patches"
    if candidate == source.strip():
        return "", applied, "unchanged_after_patch_assembly"
    return candidate, applied, ""
