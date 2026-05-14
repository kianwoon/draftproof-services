"""Contract-invariant repair layer for V3 candidates."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import compact_document_inventory, word_count


FAMILY = "contract_repair"


def build_contract_repair_prompt(
    *,
    original_text: str,
    failed_candidate: str,
    strategy_family: str,
    candidate_trace: dict[str, Any],
    compression_policy: CompressionPolicy,
) -> str:
    validation = candidate_trace.get("validation") if isinstance(candidate_trace.get("validation"), dict) else {}
    compression = candidate_trace.get("compression") if isinstance(candidate_trace.get("compression"), dict) else {}
    payload = {
        "source_document": original_text,
        "failed_candidate": failed_candidate,
        "source_word_count": word_count(original_text),
        "document_inventory": compact_document_inventory(original_text),
        "strategy_family": strategy_family,
        "failed_invariants": {
            "validation": validation,
            "compression": compression,
            "semantic_safe": bool(candidate_trace.get("semantic_safe")),
            "semantic_similarity": candidate_trace.get("semantic_similarity"),
            "external_proxy": candidate_trace.get("external_proxy"),
        },
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "must_include_exact_anchors": [
            item.get("text")
            for item in (validation.get("missing_anchors") or [])
            if isinstance(item, dict) and item.get("text")
        ],
        "requirements": [
            "Use the source document as the authority for meaning, facts, sequence, headings, citations, quotes, and protected anchors.",
            "Repair every failed invariant listed in failed_invariants before making style changes.",
            "Copy every string in must_include_exact_anchors verbatim into the repaired document.",
            "Keep each source document unit represented in the same order.",
            "Aim near preferred_words so the candidate is not a compressed summary.",
            "Do not add unsupported facts, sources, numbers, names, headings, labels, bullets, paragraph numbers, markdown, or commentary.",
            "Return only the repaired rewritten document as plain text.",
        ],
    }
    return (
        "Repair this rewrite candidate against the generic V3 contract.\n"
        "The failed candidate may have useful texture, but contract validity is mandatory.\n"
        "Fix only the listed invariant failures and keep the source document as the authority.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
