"""Cited-practice voice layer for VET and practice-grounded academic content."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v2.contracts import RewriteContract
from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import compact_document_inventory, word_count


FAMILY = "cited_practice_voice"


def _protected_anchor_payload(contract: RewriteContract, *, limit: int = 80) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    for anchor in contract.anchors[:limit]:
        anchors.append({
            "text": anchor.text,
            "severity": anchor.severity.value,
            "kind": anchor.kind,
            "section_id": anchor.section_id,
        })
    return anchors


def _chunk_anchor_payload(contract: RewriteContract, source_units: list[dict[str, Any]], *, limit: int = 80) -> list[dict[str, Any]]:
    source_text = "\n\n".join(str(unit.get("text") or unit.get("text_preview") or "") for unit in source_units)
    anchors: list[dict[str, Any]] = []
    for anchor in contract.anchors:
        if anchor.text not in source_text:
            continue
        anchors.append({
            "text": anchor.text,
            "severity": anchor.severity.value,
            "kind": anchor.kind,
            "section_id": anchor.section_id,
        })
        if len(anchors) >= limit:
            break
    return anchors


def build_cited_practice_voice_prompt(
    *,
    original_text: str,
    contract: RewriteContract,
    compression_policy: CompressionPolicy,
    style_examples: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    examples = style_examples or {"positive": [], "negative": []}
    payload = {
        "source_document": original_text,
        "source_word_count": word_count(original_text),
        "document_inventory": compact_document_inventory(original_text, max_units=120),
        "protected_anchors": _protected_anchor_payload(contract),
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "positive_external_boundaries": examples.get("positive") or [],
        "negative_external_boundaries": examples.get("negative") or [],
        "requirements": [
            "Keep the same section order and section headings.",
            "Preserve citations, unit codes, course names, named people, and support-needs details.",
            "Keep a practice-grounded educator voice where the source already supports it.",
            "Avoid polished academic summary and avoid dense over-compression.",
            "Do not add new sources, client facts, diagnoses, names, events, or examples beyond the source.",
            "Do not use bullets, markdown, paragraph numbers, or labels beyond source headings.",
            "Return only the rewritten document as plain text.",
        ],
    }
    return (
        "Rewrite this practice-grounded cited document using V3 cited-practice voice.\n"
        "Learn from any supplied external positive and negative boundaries, but do not copy them.\n"
        "Preserve source facts and protected anchors exactly.\n"
        "Return only plain text, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_cited_practice_voice_chunk_prompt(
    *,
    source_units: list[dict[str, Any]],
    contract: RewriteContract,
    global_plan: dict[str, Any],
    compression_policy: CompressionPolicy,
    style_examples: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    examples = style_examples or {"positive": [], "negative": []}
    chunk_anchors = _chunk_anchor_payload(contract, source_units)
    payload = {
        "global_plan": global_plan,
        "source_unit_count": len(source_units),
        "source_units": source_units,
        "protected_anchors": chunk_anchors,
        "must_include_exact_anchors": [anchor["text"] for anchor in chunk_anchors],
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "positive_external_boundaries": examples.get("positive") or [],
        "negative_external_boundaries": examples.get("negative") or [],
        "requirements": [
            "Rewrite only the provided source units.",
            "Keep headings and citations present in these units exactly.",
            "Copy every must_include_exact_anchors string verbatim into this chunk output.",
            "Aim near preferred_words so this chunk is not a compressed summary.",
            "Return the same number of document units as source_unit_count.",
            "Keep the practice-grounded educator voice without adding unsupported events.",
            "Return only rewritten units joined with blank lines.",
        ],
    }
    return (
        "Rewrite this cited-practice document chunk.\n"
        "Return only plain text for this chunk, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
