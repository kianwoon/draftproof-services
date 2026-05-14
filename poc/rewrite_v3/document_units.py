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


def document_units(text: str) -> list[DocumentUnit]:
    """Split a document into blank-line units and detect heading-like units.

    The heading signal is structural only: short single-line units without
    terminal sentence punctuation. It is not a keyword or domain classifier.
    """

    units: list[DocumentUnit] = []
    for index, raw in enumerate([item.strip() for item in str(text or "").split("\n\n") if item.strip()], start=1):
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        words = word_count(raw)
        is_heading = bool(
            len(lines) == 1
            and 0 < words <= 14
            and not lines[0].endswith((".", "?", "!", ":"))
        )
        units.append(DocumentUnit(unit_id=f"u{index}", text=raw, word_count=words, is_heading=is_heading))
    if not units and str(text or "").strip():
        units.append(DocumentUnit(unit_id="u1", text=str(text).strip(), word_count=word_count(text)))
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
