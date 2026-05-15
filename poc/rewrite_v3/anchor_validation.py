"""Anchor and structure validation for rewrite V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rewrite_v2.contracts import AnchorSeverity, RewriteContract, anchor_present

from .document_units import document_units, structural_shape_failures


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    failures: tuple[str, ...] = field(default_factory=tuple)
    missing_anchors: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    source_units: int = 0
    candidate_units: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _quote_normalized(value: str) -> str:
    return (
        str(value or "")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )


def _anchor_present_v3(anchor: Any, text: str) -> bool:
    if anchor.kind == "numeric" and len(str(anchor.text or "").strip()) <= 2:
        return True
    if anchor_present(anchor, text):
        return True
    if anchor.kind == "direct_quote":
        return _quote_normalized(anchor.text) in _quote_normalized(text)
    return False


def validate_v3_candidate(
    *,
    original_text: str,
    candidate_text: str,
    contract: RewriteContract,
    require_unit_count: bool = True,
    expected_unit_count: int | None = None,
) -> ValidationResult:
    failures: list[str] = []
    missing: list[dict[str, Any]] = []
    source_units = document_units(original_text)
    candidate_units = document_units(candidate_text)
    expected_units = int(expected_unit_count) if isinstance(expected_unit_count, int) and expected_unit_count > 0 else len(source_units)
    if require_unit_count and len(candidate_units) != expected_units:
        failures.append("document_unit_count_changed")
    if require_unit_count and expected_units == len(source_units):
        for failure in structural_shape_failures(original_text, candidate_text):
            if failure not in failures:
                failures.append(failure)
    for anchor in contract.anchors:
        if anchor.severity not in {AnchorSeverity.HARD_EXACT, AnchorSeverity.HARD_NORMALIZED, AnchorSeverity.TITLE_CONTEXT}:
            continue
        if not _anchor_present_v3(anchor, candidate_text):
            missing.append(anchor.to_dict())
    if missing:
        failures.append("protected_anchor_missing")
    return ValidationResult(
        passed=not failures,
        failures=tuple(failures),
        missing_anchors=tuple(missing),
        source_units=expected_units,
        candidate_units=len(candidate_units),
    )
