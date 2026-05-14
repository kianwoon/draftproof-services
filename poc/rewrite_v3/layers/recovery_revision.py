"""Second-pass recovery layer for V3 candidates that fail proxy gates."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import compact_document_inventory, word_count


FAMILY = "recovery_revision"


def build_recovery_revision_prompt(
    *,
    original_text: str,
    failed_candidate: str,
    strategy_family: str,
    proxy_feedback: dict[str, Any],
    compression_policy: CompressionPolicy,
    style_examples: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    examples = style_examples or {"positive": [], "negative": []}
    payload = {
        "source_document": original_text,
        "failed_candidate": failed_candidate,
        "source_word_count": word_count(original_text),
        "document_inventory": compact_document_inventory(original_text),
        "strategy_family": strategy_family,
        "proxy_feedback": proxy_feedback,
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "positive_external_boundaries": examples.get("positive") or [],
        "negative_external_boundaries": examples.get("negative") or [],
        "requirements": [
            "Use the source document as the authority for meaning and order.",
            "Treat the failed candidate only as evidence of what did not work.",
            "Preserve the source argument, factual claims, citations, headings, and protected anchors.",
            "Repair the proxy feedback problems directly.",
            "Do not collapse the document into a compressed summary.",
            "Do not add headings, bullets, paragraph numbers, labels, or markdown unless present in the source.",
            "Return only the rewritten document as plain text.",
        ],
    }
    return (
        "Rewrite the document again using V3 recovery revision.\n"
        "The previous candidate failed selection gates. Produce a new candidate, not commentary.\n"
        "Learn from supplied external positive and negative boundaries, but do not copy them.\n"
        "Return only plain text, no JSON and no self-check.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
