"""Plain-reasoning broad prose layer for V3."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import compact_document_inventory, word_count


FAMILY = "plain_reasoning_broad_prose"


def build_plain_reasoning_broad_prose_prompt(
    *,
    original_text: str,
    failed_candidates: list[str],
    compression_policy: CompressionPolicy,
    style_examples: dict[str, list[dict[str, Any]]],
) -> str:
    payload = {
        "source_document": original_text,
        "source_word_count": word_count(original_text),
        "document_inventory": compact_document_inventory(original_text),
        "failed_candidates": failed_candidates,
        "positive_external_boundaries": style_examples.get("positive") or [],
        "negative_external_boundaries": style_examples.get("negative") or [],
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "requirements": [
            "Rewrite as broad prose with plain reasoning, not as a formal survey.",
            "Preserve source paragraph order and source paragraph count.",
            "Preserve the source meaning, factual claims, entities, and examples.",
            "Use direct language and human judgment lines where the source supports them.",
            "Avoid textbook openings, balanced report phrasing, and polished summary transitions.",
            "Do not add unsupported facts, numbers, names, examples, headings, bullets, labels, or markdown.",
            "Aim near the preferred word count, not the minimum word count.",
            "Return only the rewritten document as plain text.",
        ],
    }
    return (
        "Rewrite this broad prose document using V3 plain-reasoning style.\n"
        "The goal is a natural argument/overview that keeps the source facts but avoids formal generated-survey texture.\n"
        "Learn from positive and negative external boundaries without copying them.\n"
        "Return only plain text, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
