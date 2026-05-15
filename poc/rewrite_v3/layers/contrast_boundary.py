"""Contrast-boundary rewrite layer for broad V3 failures."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.compression_policy import CompressionPolicy
from rewrite_v3.document_units import structural_shape_contract, word_count


FAMILY = "contrast_boundary"


def build_contrast_boundary_prompt(
    *,
    original_text: str,
    failed_candidate: str,
    family: str,
    compression_policy: CompressionPolicy,
    style_examples: dict[str, list[dict[str, Any]]],
) -> str:
    payload = {
        "source_document": original_text,
        "current_failed_rewrite": {
            "failure": "candidate still reads like a clean generated essay or detector-visible formal survey",
            "text": failed_candidate,
        },
        "positive_boundary_samples": style_examples.get("positive") or [],
        "negative_boundary_samples": style_examples.get("negative") or [],
        "source_word_count": word_count(original_text),
        "source_structure_contract": structural_shape_contract(original_text),
        "target_word_band": {
            "min_words": compression_policy.min_words,
            "preferred_words": compression_policy.preferred_words,
            "max_words": compression_policy.max_words,
        },
        "strategy_family": family,
        "instructions": [
            "Preserve the source meaning and paragraph sequence.",
            "Do not add unsupported facts, names, examples, or claims.",
            "Do not produce a neat balanced survey in every paragraph.",
            "Use plain reasoning turns learned from positive boundaries, without copying them.",
            "Avoid the formal texture shown in failed and negative samples.",
            "Keep paragraph count aligned with the source when the source is paragraph-based.",
            "Preserve source_structure_contract exactly: same block_count, same blank_line_boundary_count, and same heading_like_lines.",
            "Do not add blank-line paragraph splits or merge source blocks.",
        ],
        "output_schema": {
            "rewritten_document": "plain text with paragraphs separated by blank lines",
            "notes": ["short notes"],
        },
    }
    return (
        "Rewrite the document as a V3 contrast-boundary pass.\n"
        "Infer what failed from the failed rewrite and negative boundaries, then write a fresh candidate.\n"
        "Return JSON only with rewritten_document and notes.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def extract_contrast_boundary_output(raw: str) -> str:
    text = str(raw or "").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return text
        else:
            return text
    if isinstance(parsed, dict):
        value = parsed.get("rewritten_document")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return text
