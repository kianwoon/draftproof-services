"""Document unit helpers for rewrite V3.

These helpers intentionally use simple structural parsing rather than keyword
matching. V3 should route from scan contracts and document shape, not ad hoc
phrase rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentUnit:
    unit_id: str
    text: str
    word_count: int
    is_heading: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def word_count(text: str) -> int:
    return len(str(text or "").split())


def is_heading_like_line(line: str) -> bool:
    stripped = str(line or "").strip()
    return bool(
        stripped
        and "\n" not in stripped
        and 0 < word_count(stripped) <= 14
        and not stripped.endswith((".", "?", "!", ":"))
    )


def structural_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in str(text or "").splitlines():
        if not line.strip():
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line.rstrip())
    if current:
        blocks.append("\n".join(current).strip())
    return blocks


def structural_shape_contract(text: str) -> dict[str, Any]:
    blocks = structural_blocks(text)
    heading_lines: list[str] = []
    nonempty_lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in nonempty_lines:
        if is_heading_like_line(line):
            heading_lines.append(line)
    return {
        "block_count": len(blocks),
        "blank_line_boundary_count": max(0, len(blocks) - 1),
        "heading_like_line_count": len(heading_lines),
        "heading_like_lines": heading_lines,
        "first_nonempty_line": nonempty_lines[0] if nonempty_lines else "",
        "last_nonempty_line": nonempty_lines[-1] if nonempty_lines else "",
    }


def structural_shape_contract_for_units(source_units: list[dict[str, Any]]) -> dict[str, Any]:
    unit_texts = [
        str(unit.get("text") or unit.get("text_preview") or "").strip()
        for unit in source_units or []
        if str(unit.get("text") or unit.get("text_preview") or "").strip()
    ]
    shape = structural_shape_contract("\n\n".join(unit_texts))
    shape["source_unit_count"] = len(unit_texts)
    return shape


def structural_shape_failures(source_text: str, replacement_text: str) -> list[str]:
    source_shape = structural_shape_contract(source_text)
    replacement_shape = structural_shape_contract(replacement_text)
    failures: list[str] = []
    if replacement_shape["block_count"] != source_shape["block_count"]:
        failures.append("block_count_changed")
    if replacement_shape["blank_line_boundary_count"] != source_shape["blank_line_boundary_count"]:
        failures.append("blank_line_boundary_count_changed")
    missing_headings = [
        heading
        for heading in source_shape["heading_like_lines"]
        if heading not in replacement_shape["heading_like_lines"]
    ]
    if missing_headings:
        failures.append("heading_like_line_missing")
    return failures


def document_units(text: str) -> list[DocumentUnit]:
    """Split a document into blank-line units and detect heading-like units.

    The heading signal is structural only: short single-line units without
    terminal sentence punctuation. It is not a keyword or domain classifier.
    """

    raw_units: list[DocumentUnit] = []
    for index, raw in enumerate(structural_blocks(text), start=1):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        words = word_count(raw)
        is_heading = bool(
            len(lines) == 1
            and is_heading_like_line(lines[0])
        )
        raw_units.append(DocumentUnit(unit_id=f"u{index}", text=raw, word_count=words, is_heading=is_heading))
    if not raw_units and str(text or "").strip():
        raw_units.append(DocumentUnit(unit_id="u1", text=str(text).strip(), word_count=word_count(text)))

    units: list[DocumentUnit] = []
    pending_heading: DocumentUnit | None = None
    for unit in raw_units:
        if unit.is_heading:
            if pending_heading is not None:
                units.append(pending_heading)
            pending_heading = unit
            continue
        if pending_heading is not None:
            merged_text = f"{pending_heading.text}\n{unit.text}".strip()
            units.append(DocumentUnit(
                unit_id=f"u{len(units) + 1}",
                text=merged_text,
                word_count=word_count(merged_text),
                is_heading=False,
            ))
            pending_heading = None
        else:
            units.append(DocumentUnit(
                unit_id=f"u{len(units) + 1}",
                text=unit.text,
                word_count=unit.word_count,
                is_heading=False,
            ))
    if pending_heading is not None:
        units.append(DocumentUnit(
            unit_id=f"u{len(units) + 1}",
            text=pending_heading.text,
            word_count=pending_heading.word_count,
            is_heading=True,
        ))
    return units


def compose_units(units: list[str]) -> str:
    return "\n\n".join(str(unit or "").strip() for unit in units if str(unit or "").strip())


def compact_document_inventory(text: str, *, max_units: int = 80, preview_chars: int = 520) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for unit in document_units(text)[:max_units]:
        preview = unit.text
        if len(preview) > preview_chars:
            preview = preview[:preview_chars].rstrip() + "..."
        inventory.append({
            "unit_id": unit.unit_id,
            "word_count": unit.word_count,
            "is_heading": unit.is_heading,
            "text_preview": preview,
        })
    return inventory
