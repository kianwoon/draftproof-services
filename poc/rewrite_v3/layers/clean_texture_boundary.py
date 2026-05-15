"""Clean texture boundary rewrite layer for broad V3 prose."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.document_units import compact_document_inventory, word_count
from rewrite_v3.prompt_contract import profile_action_contracts


FAMILY = "clean_texture_boundary"


def _scan_problem_profile(scan_report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(scan_report, dict):
        return {}
    scan_intelligence = scan_report.get("scan_intelligence") if isinstance(scan_report.get("scan_intelligence"), dict) else {}
    footprint_gate = scan_intelligence.get("ai_footprint_gate") if isinstance(scan_intelligence.get("ai_footprint_gate"), dict) else {}
    turnitin_gate = scan_intelligence.get("turnitin_like_gate") if isinstance(scan_intelligence.get("turnitin_like_gate"), dict) else {}
    span_gate = scan_intelligence.get("eligible_span_density_gate") if isinstance(scan_intelligence.get("eligible_span_density_gate"), dict) else {}
    return {
        "authorship_footprint": (footprint_gate.get("before") or {}).get("authorship_footprint"),
        "semantic_footprint": (footprint_gate.get("before") or {}).get("semantic_footprint"),
        "structural_footprint": (footprint_gate.get("before") or {}).get("structural_footprint"),
        "external_ai_flag_risk": (footprint_gate.get("before") or {}).get("external_ai_flag_risk"),
        "turnitin_like_components": (turnitin_gate.get("before") or {}).get("components"),
        "unsafe_clusters": (span_gate.get("top_unsafe_clusters") or [])[:4],
        "sentence_targets": (span_gate.get("top_sentence_targets") or [])[:12],
    }


def build_clean_texture_boundary_prompt(
    *,
    original_text: str,
    scan_report: dict[str, Any] | None,
    style_examples: dict[str, list[dict[str, Any]]] | None = None,
    rewrite_target_profile: dict[str, Any] | None = None,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    central_judgment_plan: dict[str, Any] | None = None,
) -> str:
    examples = style_examples or {"positive": [], "negative": []}
    payload = {
        "source_document": original_text,
        "source_word_count": word_count(original_text),
        "document_inventory": compact_document_inventory(original_text),
        "scanner_problem_profile": _scan_problem_profile(scan_report),
        "rewrite_target_profile": rewrite_target_profile or {},
        "scanner_action_contracts": profile_action_contracts(
            rewrite_target_profile=rewrite_target_profile,
            predictability_briefs=predictability_briefs,
            compact=True,
        ),
        "central_judgment_plan": central_judgment_plan or {},
        "positive_external_boundaries": examples.get("positive") or [],
        "negative_external_boundaries": examples.get("negative") or [],
        "objective": [
            "Reduce scanner-visible authorship texture and predictable sentence paths.",
            "Use rewrite_target_profile targets as the primary scanner-derived repair contract when present.",
            "Use scanner_action_contracts for exact target operations and predictable spans when present.",
            "Use central_judgment_plan to choose contextual anchors, reasoning turns, and non-formulaic judgment.",
            "Replace formal survey rhythm with clean natural reasoning.",
            "Keep the writing grammatical, readable, and suitable for the source content.",
            "Preserve source facts, paragraph order, entities, numbers, examples, and claims.",
        ],
        "hard_limits": [
            "Do not optimize for word count or follow a word band.",
            "Do not add unsupported facts, names, numbers, examples, citations, headings, bullets, labels, or markdown.",
            "Do not use artificial roughness, fragments, ellipses as style devices, or deliberate errors.",
            "Do not mention detectors, AI, rewriting, prompts, scores, or these instructions.",
            "Return only the rewritten document with blank lines between paragraphs.",
        ],
    }
    return (
        "Rewrite the document using V3 clean-texture boundary reconstruction.\n"
        "Use the scanner problem profile to change sentence path and rhythm where the source reads like generated survey prose.\n"
        "Length is not the objective; AI-texture mitigation and clean readability are the objective.\n"
        "Learn from supplied external boundaries when present, but do not copy them.\n"
        "Return only plain text, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_clean_texture_boundary_chunk_prompt(
    *,
    source_units: list[dict[str, Any]],
    global_plan: dict[str, Any],
    style_examples: dict[str, list[dict[str, Any]]] | None = None,
    rewrite_target_profile: dict[str, Any] | None = None,
    predictability_briefs: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    central_judgment_plan: dict[str, Any] | None = None,
) -> str:
    examples = style_examples or {"positive": [], "negative": []}
    payload = {
        "global_plan": global_plan,
        "source_units": source_units,
        "rewrite_target_profile": rewrite_target_profile or {},
        "scanner_action_contracts": profile_action_contracts(
            rewrite_target_profile=rewrite_target_profile,
            predictability_briefs=predictability_briefs,
            compact=True,
        ),
        "central_judgment_plan": central_judgment_plan or {},
        "positive_external_boundaries": examples.get("positive") or [],
        "negative_external_boundaries": examples.get("negative") or [],
        "objective": [
            "Rewrite only the provided source units.",
            "Use rewrite_target_profile targets for this chunk when they overlap these units.",
            "Use scanner_action_contracts for exact target operations and predictable spans when present.",
            "Use central_judgment_plan operations only where they fit the source units.",
            "Reduce formal survey texture and predictable sentence paths inside this chunk.",
            "Keep unit order and preserve local meaning, facts, entities, numbers, examples, and claims.",
            "Keep clean readable prose; do not create artificial roughness.",
        ],
        "hard_limits": [
            "Do not optimize for word count or follow a word band.",
            "Do not add unsupported facts, names, numbers, examples, citations, headings, bullets, labels, or markdown.",
            "Do not mention detectors, AI, rewriting, prompts, scores, or these instructions.",
            "Return only rewritten units joined with blank lines.",
        ],
    }
    return (
        "Rewrite this document chunk using V3 clean-texture boundary reconstruction.\n"
        "Keep the chunk compatible with the surrounding document, but change generated-survey sentence paths.\n"
        "Return only plain text for this chunk, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
