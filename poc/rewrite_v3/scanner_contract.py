"""Typed scan contract for rewrite V3 routing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .document_units import document_units, word_count


class RewriteRiskClass(str, Enum):
    BROAD_PROSE = "broad_prose"
    CITED_ACADEMIC = "cited_academic"
    TECHNICAL_STRUCTURED = "technical_structured"
    REGULATED_POLICY = "regulated_policy"
    QUOTE_OR_EVIDENCE_HEAVY = "quote_or_evidence_heavy"
    PERSONAL_REFLECTIVE = "personal_reflective"
    CREATIVE_MARKETING = "creative_marketing"
    SHORT_OR_SPARSE = "short_or_sparse"


@dataclass(frozen=True)
class ScanContract:
    word_count: int
    unit_count: int
    heading_count: int
    avg_unit_words: float
    ai_score: float | None
    topk_score: float | None
    writing_quality_score: float | None
    content_mode: str
    content_mode_confidence: float
    mode_scores: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    hard_anchor_count: int = 0
    citation_anchor_count: int = 0
    quote_anchor_count: int = 0
    findings_count: int = 0
    rewrite_brief_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _first_route(scan_report: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        scan_report.get("rewrite_v2_content_route"),
        scan_report.get("content_route"),
        ((scan_report.get("scan_intelligence") or {}).get("mitigation_inputs") or {}).get("rewrite_v2_content_route"),
        ((scan_report.get("scan_intelligence") or {}).get("generation_handoff") or {}).get("rewrite_v2_content_route"),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return candidate
    return {}


def _anchor_counts(scan_report: dict[str, Any]) -> tuple[int, int, int]:
    inventory = (((scan_report.get("scan_intelligence") or {}).get("document") or {}).get("preservation_inventory") or [])
    hard = 0
    citations = 0
    quotes = 0
    if isinstance(inventory, list):
        for item in inventory:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or item.get("type") or "")
            severity = str(item.get("severity") or "")
            if severity.startswith("hard"):
                hard += 1
            if "citation" in kind:
                citations += 1
            if "quote" in kind:
                quotes += 1
    return hard, citations, quotes


def _findings_count(scan_report: dict[str, Any]) -> int:
    findings = scan_report.get("findings")
    if not isinstance(findings, dict):
        return 0
    return sum(len(value) for value in findings.values() if isinstance(value, list))


def build_scan_contract(scan_report: dict[str, Any], original_text: str) -> ScanContract:
    units = document_units(original_text)
    route = _first_route(scan_report)
    badge = scan_report.get("ai_risk_badge") if isinstance(scan_report.get("ai_risk_badge"), dict) else {}
    components = badge.get("ai_components") if isinstance(badge.get("ai_components"), dict) else {}
    hard, citations, quotes = _anchor_counts(scan_report)
    mode_scores = route.get("mode_scores") if isinstance(route.get("mode_scores"), list) else []
    return ScanContract(
        word_count=word_count(original_text),
        unit_count=len(units),
        heading_count=sum(1 for unit in units if unit.is_heading),
        avg_unit_words=round(sum(unit.word_count for unit in units) / max(1, len(units)), 2),
        ai_score=_float_or_none(badge.get("ai_likelihood_score")),
        topk_score=_float_or_none(components.get("topk_pattern_raw")),
        writing_quality_score=_float_or_none(badge.get("writing_quality_score")),
        content_mode=str(route.get("primary_mode") or route.get("content_mode") or "unknown"),
        content_mode_confidence=float(route.get("confidence") or route.get("content_mode_confidence") or 0.0),
        mode_scores=tuple(row for row in mode_scores if isinstance(row, dict)),
        hard_anchor_count=hard,
        citation_anchor_count=citations,
        quote_anchor_count=quotes,
        findings_count=_findings_count(scan_report),
        rewrite_brief_count=len(scan_report.get("rewrite_edit_briefs") or []),
    )
