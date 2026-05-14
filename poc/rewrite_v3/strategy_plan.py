"""Strategy stack builder for rewrite V3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .router import V3Route
from .scanner_contract import RewriteRiskClass, ScanContract


@dataclass(frozen=True)
class StrategyStep:
    strategy_id: str
    target_issue: str
    editable_scope: str
    max_candidates: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyPlan:
    risk_classes: tuple[str, ...]
    steps: tuple[StrategyStep, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_classes": list(self.risk_classes),
            "steps": [step.to_dict() for step in self.steps],
        }


def build_strategy_plan(route: V3Route, contract: ScanContract) -> StrategyPlan:
    classes = (route.primary_class, *route.secondary_classes)
    steps: list[StrategyStep] = []
    if RewriteRiskClass.BROAD_PROSE in classes:
        steps.extend([
            StrategyStep("document_rhythm", "broad_ai_rhythm", "full_document"),
            StrategyStep("contrast_boundary", "formal_generated_texture", "full_document"),
            StrategyStep("plain_reasoning_broad_prose", "formal_survey_texture", "full_document"),
            StrategyStep("topk_texture_repair", "topk_residual", "candidate_revision"),
        ])
    if RewriteRiskClass.CITED_ACADEMIC in classes:
        steps.extend([
            StrategyStep("citation_anchor_guard", "citation_or_anchor_loss", "protected_anchors"),
            StrategyStep("cited_practice_voice", "academic_detector_texture", "section_or_document"),
        ])
    if RewriteRiskClass.TECHNICAL_STRUCTURED in classes:
        steps.append(StrategyStep("structure_preserving_rewrite", "technical_structure_risk", "minimal_targeted"))
    if RewriteRiskClass.REGULATED_POLICY in classes:
        steps.append(StrategyStep("obligation_preserving_patch", "regulated_claim_risk", "minimal_targeted"))
    if RewriteRiskClass.QUOTE_OR_EVIDENCE_HEAVY in classes:
        steps.append(StrategyStep("quote_preservation_guard", "quote_or_evidence_loss", "surrounding_analysis_only"))
    if RewriteRiskClass.PERSONAL_REFLECTIVE in classes:
        steps.append(StrategyStep("voice_preservation", "personal_voice_flattening", "paragraph_or_section"))
    if RewriteRiskClass.CREATIVE_MARKETING in classes:
        steps.append(StrategyStep("genre_specific_rewrite", "marketing_style_texture", "document_or_block"))
    if RewriteRiskClass.SHORT_OR_SPARSE in classes:
        steps.append(StrategyStep("limited_context_expansion", "insufficient_context", "bounded_expansion"))
    steps.append(StrategyStep("portfolio_selection", "candidate_selection", "candidate_set"))
    if contract.hard_anchor_count > 0:
        steps.append(StrategyStep("anchor_repair", "protected_anchor_loss", "exact_anchor_restoration"))
    steps.append(StrategyStep("structure_repair", "document_structure_changed", "candidate_boundaries"))
    unique: list[StrategyStep] = []
    seen: set[str] = set()
    for step in steps:
        if step.strategy_id in seen:
            continue
        seen.add(step.strategy_id)
        unique.append(step)
    return StrategyPlan(
        risk_classes=tuple(item.value for item in classes),
        steps=tuple(unique),
    )
