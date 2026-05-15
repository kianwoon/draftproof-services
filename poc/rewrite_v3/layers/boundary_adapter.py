"""External-boundary adapter layer for V3."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import compact_document_inventory, structural_shape_contract, word_count


FAMILY = "boundary_adapter"


def build_boundary_adapter_prompt(
    *,
    original_text: str,
    failed_candidates: list[str],
    strategy_family: str,
    proxy_feedback: list[dict[str, Any]],
    compression_policy: CompressionPolicy,
    style_examples: dict[str, list[dict[str, Any]]],
) -> str:
    payload = {
        "source_document": original_text,
        "failed_candidates": failed_candidates,
        "source_word_count": word_count(original_text),
        "document_inventory": compact_document_inventory(original_text),
        "source_structure_contract": structural_shape_contract(original_text),
        "strategy_family": strategy_family,
        "proxy_feedback": proxy_feedback,
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "positive_external_boundaries": style_examples.get("positive") or [],
        "negative_external_boundaries": style_examples.get("negative") or [],
        "requirements": [
            "Use the source document as the authority for meaning, paragraph order, and factual claims.",
            "Use positive external boundaries as style/rhythm targets.",
            "Avoid the patterns shown by negative boundaries and failed candidates.",
            "For paragraph-based prose, preserve the same paragraph count as the source.",
            "Keep each source paragraph represented by one rewritten paragraph in the same position.",
            "Preserve source_structure_contract exactly: same block_count, same blank_line_boundary_count, and same heading_like_lines.",
            "Do not add blank-line paragraph splits or merge source blocks.",
            "Aim near the preferred word count, not the minimum word count.",
            "Do not add unsupported facts, citations, people, places, numbers, or anecdotes.",
            "Do not add headings, bullets, paragraph numbers, labels, markdown, or commentary.",
            "Return only the rewritten document as plain text.",
        ],
    }
    return (
        "Create a new rewrite using V3 external-boundary adaptation.\n"
        "The earlier candidates failed. Do not repair them line by line; write a fresh version.\n"
        "The positive boundaries show the target rhythm. The source document controls meaning.\n"
        "Return only plain text, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
