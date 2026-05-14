"""Scanner-contract router for rewrite V3 risk classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .scanner_contract import RewriteRiskClass, ScanContract


@dataclass(frozen=True)
class V3Route:
    primary_class: RewriteRiskClass
    secondary_classes: tuple[RewriteRiskClass, ...]
    confidence: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_class"] = self.primary_class.value
        payload["secondary_classes"] = [item.value for item in self.secondary_classes]
        return payload


def _score_classes(contract: ScanContract) -> dict[RewriteRiskClass, float]:
    scores = {risk_class: 0.0 for risk_class in RewriteRiskClass}
    mode = contract.content_mode
    if mode in {"broad_explanatory_essay", "generic_expository"}:
        scores[RewriteRiskClass.BROAD_PROSE] += 0.65
    if mode == "academic_cited_text":
        scores[RewriteRiskClass.CITED_ACADEMIC] += 0.8
    if mode == "technical_content":
        scores[RewriteRiskClass.TECHNICAL_STRUCTURED] += 0.8
    if mode == "regulated_policy_content":
        scores[RewriteRiskClass.REGULATED_POLICY] += 0.8
    if mode == "quote_heavy":
        scores[RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY] += 0.8
    if mode == "personal_reflection":
        scores[RewriteRiskClass.PERSONAL_REFLECTIVE] += 0.8
    if mode == "creative_marketing":
        scores[RewriteRiskClass.CREATIVE_MARKETING] += 0.8
    if mode == "short_text":
        scores[RewriteRiskClass.SHORT_OR_SPARSE] += 0.8

    if contract.word_count <= 120 or contract.unit_count <= 1:
        scores[RewriteRiskClass.SHORT_OR_SPARSE] += 0.35
    if contract.citation_anchor_count > 0:
        scores[RewriteRiskClass.CITED_ACADEMIC] += min(0.7, 0.36 + contract.citation_anchor_count * 0.08)
    if contract.quote_anchor_count > 0:
        scores[RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY] += min(0.6, 0.28 + contract.quote_anchor_count * 0.1)
        scores[RewriteRiskClass.CITED_ACADEMIC] += min(0.24, contract.quote_anchor_count * 0.06)
    if contract.heading_count >= 2 and contract.avg_unit_words <= 90:
        scores[RewriteRiskClass.TECHNICAL_STRUCTURED] += 0.18
    if contract.hard_anchor_count >= 6:
        scores[RewriteRiskClass.CITED_ACADEMIC] += 0.12
        scores[RewriteRiskClass.TECHNICAL_STRUCTURED] += 0.08
    if contract.unit_count >= 4 and contract.word_count > 300:
        scores[RewriteRiskClass.BROAD_PROSE] += 0.18
    return scores


def route_from_scan_contract(contract: ScanContract) -> V3Route:
    scores = _score_classes(contract)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary, primary_score = ordered[0]
    secondary = tuple(risk_class for risk_class, score in ordered[1:] if score >= 0.28)
    reasons = [
        f"scan_content_mode={contract.content_mode}",
        f"units={contract.unit_count}",
        f"words={contract.word_count}",
        f"hard_anchors={contract.hard_anchor_count}",
        f"citations={contract.citation_anchor_count}",
        f"quotes={contract.quote_anchor_count}",
    ]
    return V3Route(
        primary_class=primary,
        secondary_classes=secondary[:4],
        confidence=round(min(0.98, max(0.35, primary_score)), 3),
        reasons=tuple(reasons),
    )
