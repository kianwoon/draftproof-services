"""Document-rhythm reconstruction layer for broad expository content."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import compact_document_inventory, word_count
from rewrite_v3.prompt_contract import profile_action_contracts


FAMILY = "document_rhythm"


def build_document_rhythm_prompt(
    *,
    original_text: str,
    compression_policy: CompressionPolicy,
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
        "rewrite_target_profile": rewrite_target_profile or {},
        "scanner_action_contracts": profile_action_contracts(
            rewrite_target_profile=rewrite_target_profile,
            predictability_briefs=predictability_briefs,
            compact=True,
        ),
        "central_judgment_plan": central_judgment_plan or {},
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "positive_external_boundaries": examples.get("positive") or [],
        "negative_external_boundaries": examples.get("negative") or [],
        "requirements": [
            "Preserve the source argument and paragraph order.",
            "Use rewrite_target_profile targets as the primary rewrite instructions when present.",
            "Use scanner_action_contracts for exact target operations and predictable spans when present.",
            "For each target, address dominant_drivers and required_movement without compressing meaning.",
            "Use central_judgment_plan to add source-supported contextual reasoning and avoid formulaic survey closure.",
            "Use natural document rhythm instead of a uniformly balanced essay structure.",
            "Allow controlled compression, but do not collapse the document into a dense summary.",
            "Do not add unsupported facts, examples, names, or claims.",
            "Do not use headings, bullets, paragraph numbers, labels, or markdown unless present in the source.",
            "Return only the rewritten document as plain text.",
        ],
    }
    return (
        "Rewrite this document using V3 document-rhythm reconstruction.\n"
        "Learn from any supplied external positive and negative boundaries, but do not copy them.\n"
        "The output should preserve meaning while avoiding a clean generated essay rhythm.\n"
        "Return only plain text, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_document_rhythm_chunk_prompt(
    *,
    source_units: list[dict[str, Any]],
    global_plan: dict[str, Any],
    compression_policy: CompressionPolicy,
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
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "positive_external_boundaries": examples.get("positive") or [],
        "negative_external_boundaries": examples.get("negative") or [],
        "requirements": [
            "Rewrite only the provided source units.",
            "Keep unit order and do not introduce facts outside these units.",
            "Use rewrite_target_profile targets for this chunk when they overlap these units.",
            "Use scanner_action_contracts for exact target operations and predictable spans when present.",
            "Use central_judgment_plan operations only where they fit these units.",
            "Use the global document rhythm plan, but preserve local meaning.",
            "Return only rewritten units joined with blank lines.",
        ],
    }
    return (
        "Rewrite this document chunk using V3 document-rhythm reconstruction.\n"
        "Return only plain text for this chunk, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
