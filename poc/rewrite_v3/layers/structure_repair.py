"""Structure-only repair layer for V3 candidates."""

from __future__ import annotations

import json
from typing import Any

from rewrite_v3.document_units import document_units


FAMILY = "structure_repair"


def build_structure_repair_prompt(
    *,
    source_text: str,
    candidate_text: str,
    validation: dict[str, Any],
) -> str:
    payload = {
        "source_unit_count": len(document_units(source_text)),
        "candidate_unit_count": len(document_units(candidate_text)),
        "source_units": [unit.to_dict() for unit in document_units(source_text)],
        "candidate_text": candidate_text,
        "validation": validation,
        "requirements": [
            "Repair only paragraph or document-unit boundaries.",
            "Do not rewrite sentences.",
            "Do not add new claims, examples, headings, labels, bullets, or paragraph numbers.",
            "Preserve the candidate wording as much as possible.",
            "Return exactly the source_unit_count document units separated by blank lines.",
            "Return only the repaired text.",
        ],
    }
    return (
        "Repair the structure of this candidate without changing its style.\n"
        "The candidate already has the desired writing texture, but the document-unit count is wrong.\n"
        "Only adjust blank-line boundaries so it aligns with the source structure.\n\n"
        f"PAYLOAD:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
