"""Structure-only repair layer for V3 candidates."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.document_units import document_units, structural_shape_contract


FAMILY = "structure_repair"


def build_structure_repair_prompt(
    *,
    source_text: str,
    candidate_text: str,
    validation: dict[str, Any],
    expected_unit_count: int | None = None,
) -> str:
    source_units = document_units(source_text)
    repair_unit_count = (
        int(expected_unit_count)
        if isinstance(expected_unit_count, int) and expected_unit_count > 0
        else len(source_units)
    )
    payload = {
        "source_unit_count": repair_unit_count,
        "structural_source_unit_count": len(source_units),
        "candidate_unit_count": len(document_units(candidate_text)),
        "candidate_word_count": len(str(candidate_text or "").split()),
        "source_units": [unit.to_dict() for unit in source_units],
        "source_structure_contract": structural_shape_contract(source_text),
        "candidate_text": candidate_text,
        "validation": validation,
        "requirements": [
            "Repair only paragraph or document-unit boundaries.",
            "Do not rewrite sentences.",
            "Do not remove, summarize, compress, or replace candidate wording.",
            "Do not add new claims, examples, headings, labels, bullets, or paragraph numbers.",
            "Preserve the candidate wording as much as possible.",
            "Return exactly the source_unit_count document units separated by blank lines.",
            "Preserve source_structure_contract.heading_like_lines exactly.",
            "If structural_source_unit_count differs from source_unit_count, preserve logical scan-window boundaries rather than flattening the candidate.",
            "The returned word count should stay close to candidate_word_count because this is boundary repair only.",
            "Return only the repaired text.",
        ],
    }
    return (
        "Repair the structure of this candidate without changing its style.\n"
        "The candidate already has the desired writing texture, but the document-unit count is wrong.\n"
        "Only adjust blank-line boundaries so it aligns with the source structure.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
