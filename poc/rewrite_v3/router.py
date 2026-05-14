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
    footprint_total = max(0.0, min(1.0, contract.footprint_fraction_ai + contract.footprint_fraction_ai_assisted))
    localized_footprint = (
        contract.risky_window_density > 0.0
        and contract.risky_window_density <= 0.28
        and contract.max_risky_window_words <= 160
        and contract.risky_window_count <= max(2, contract.unit_count // 3)
    )
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
    citation_signal = max(contract.citation_count, contract.citation_key_count, contract.reference_count)
    if citation_signal > 0:
        scores[RewriteRiskClass.CITED_ACADEMIC] += min(0.7, 0.36 + citation_signal * 0.08)
    if citation_signal > 0 and contract.anchor_preservation_pressure >= 0.5:
        scores[RewriteRiskClass.CITED_ACADEMIC] += 0.18
    if contract.evidence_anchor_score >= 0.5:
        scores[RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY] += 0.45
    direct_quote_count = int((contract.quote_role_counts or {}).get("direct_quote") or 0)
    if direct_quote_count >= 2:
        scores[RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY] += 0.25
    if contract.quote_count > 0 and contract.evidence_anchor_score < 0.3:
        scores[RewriteRiskClass.BROAD_PROSE] += 0.12
    if contract.heading_count >= 2 and contract.avg_unit_words <= 90:
        scores[RewriteRiskClass.TECHNICAL_STRUCTURED] += 0.18
    if contract.hard_anchor_count >= 6:
        scores[RewriteRiskClass.CITED_ACADEMIC] += 0.12
        scores[RewriteRiskClass.TECHNICAL_STRUCTURED] += 0.08
    if contract.unit_count >= 4 and contract.word_count > 300:
        scores[RewriteRiskClass.BROAD_PROSE] += 0.18
    if footprint_total >= 0.55 and contract.unit_count >= 4:
        scores[RewriteRiskClass.BROAD_PROSE] += 0.24
    if contract.rewrite_targets:
        operations = contract.target_operation_mix or {}
        if operations.get("protected_section_rewrite") or operations.get("citation_preserving_window_repair"):
            scores[RewriteRiskClass.CITED_ACADEMIC] += 0.22
        if operations.get("chunk_reconstruction"):
            scores[RewriteRiskClass.BROAD_PROSE] += 0.2
        if operations.get("grounded_author_reasoning_rewrite") or operations.get("light_texture_rewrite"):
            scores[RewriteRiskClass.BROAD_PROSE] += 0.08
    if contract.target_scope_policy == "avoid_over_rewrite":
        scores[RewriteRiskClass.BROAD_PROSE] *= 0.55
        scores[RewriteRiskClass.CITED_ACADEMIC] *= 0.65
    if localized_footprint and citation_signal > 0:
        scores[RewriteRiskClass.CITED_ACADEMIC] += 0.16
    elif localized_footprint:
        scores[RewriteRiskClass.BROAD_PROSE] += 0.10
    if footprint_total < 0.25 and contract.high_confidence_risky_window_count == 0:
        scores[RewriteRiskClass.BROAD_PROSE] *= 0.75
        scores[RewriteRiskClass.CITED_ACADEMIC] *= 0.85
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
        f"citations={contract.citation_count}",
        f"citation_keys={contract.citation_key_count}",
        f"quotes={contract.quote_count}",
        f"evidence_anchor_score={contract.evidence_anchor_score}",
        f"anchor_preservation_pressure={contract.anchor_preservation_pressure}",
        f"footprint_ai={contract.footprint_fraction_ai}",
        f"footprint_ai_assisted={contract.footprint_fraction_ai_assisted}",
        f"footprint_human={contract.footprint_fraction_human}",
        f"risky_window_density={contract.risky_window_density}",
        f"max_risky_window_words={contract.max_risky_window_words}",
        f"high_confidence_risky_windows={contract.high_confidence_risky_window_count}",
        f"rewrite_targets={len(contract.rewrite_targets)}",
        f"target_scope_policy={contract.target_scope_policy}",
        f"target_operation_mix={contract.target_operation_mix}",
    ]
    return V3Route(
        primary_class=primary,
        secondary_classes=secondary[:4],
        confidence=round(min(0.98, max(0.35, primary_score)), 3),
        reasons=tuple(reasons),
    )
