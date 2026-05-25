from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .scan import Finding, Scan, findings_for_paragraph, select_target_paragraph
from .text import Paragraph, source_terms


@dataclass(frozen=True)
class PlanAction:
    sentence_id: str
    tags: list[str]
    operation: str
    method: str
    preserve_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Plan:
    paragraph_id: str
    route_goal: str
    opening_terms: list[str]
    actions: list[PlanAction]

    def to_dict(self) -> dict[str, Any]:
        return {
            "paragraph_id": self.paragraph_id,
            "route_goal": self.route_goal,
            "opening_terms": list(self.opening_terms),
            "actions": [action.to_dict() for action in self.actions],
        }


def build_plan(scan: Scan, excluded_paragraph_ids: set[str] | None = None) -> tuple[Paragraph, Plan]:
    paragraph = select_target_paragraph(scan, excluded_paragraph_ids)
    findings = {finding.sentence_id: finding for finding in findings_for_paragraph(scan, paragraph.id)}
    actions: list[PlanAction] = []
    for sentence in paragraph.sentences:
        finding = findings.get(sentence.id)
        tags = finding.tags if finding else []
        actions.append(
            PlanAction(
                sentence_id=sentence.id,
                tags=tags,
                operation=_operation(tags),
                method=_method(tags),
                preserve_terms=source_terms(sentence.text, limit=8),
            )
        )
    return paragraph, Plan(
        paragraph_id=paragraph.id,
        route_goal="change sentence route while preserving source meaning and reducing packed or predictable shape",
        opening_terms=source_terms(paragraph.sentences[0].text if paragraph.sentences else paragraph.text, limit=6),
        actions=actions,
    )


def _operation(tags: list[str]) -> str:
    if "author_anchor_gap" in tags:
        return "add a reviewable author-proxy bridge grounded in submitted context"
    if "unsupported_claim_gap" in tags:
        return "add source-supported support, narrow the claim, or mark reviewable author-proxy provenance"
    if "context_anchor_gap" in tags:
        return "replace unsupported demonstrative route with a source-grounded context bridge"
    if "semantic_bridge_gap" in tags:
        return "make the missing source-to-claim reasoning bridge explicit"
    if "citation_anchor" in tags:
        return "keep source or citation relation close to the claim while reducing packed shape"
    if "broad_claim" in tags:
        return "narrow broad claim using source terms or mark reviewable bridge provenance"
    if "transition_stack" in tags:
        return "remove stacked transition and attach the sentence to a source beat"
    if "paraphrase_smoothing" in tags:
        return "replace smooth paraphrase wrapper with source-level sentence pressure"
    if "packed_list" in tags:
        return "unpack packed list into shorter source-preserving clauses or sentences"
    if "sentence_overload" in tags:
        return "split overloaded sentence without removing source ideas"
    if "predictable_start" in tags:
        return "change the opener using source-derived terms"
    if "abstract_density" in tags:
        return "replace abstract wrapping with concrete source terms"
    return "preserve meaning while varying sentence route"


def _method(tags: list[str]) -> str:
    if "author_anchor_gap" in tags:
        return "author_proxy_bridge"
    if "unsupported_claim_gap" in tags:
        return "author_proxy_bridge"
    if "context_anchor_gap" in tags:
        return "context_anchor_bridge"
    if "semantic_bridge_gap" in tags:
        return "semantic_bridge_repair"
    if "citation_anchor" in tags:
        return "citation_relation_repair"
    if "broad_claim" in tags:
        return "claim_scope_repair"
    if "transition_stack" in tags:
        return "transition_rebuild"
    if "paraphrase_smoothing" in tags:
        return "source_revoice"
    if "packed_list" in tags or "sentence_overload" in tags:
        return "atomic_decomposition"
    if "predictable_start" in tags or "paragraph_rhythm" in tags:
        return "route_rebuild"
    return "preserve"
