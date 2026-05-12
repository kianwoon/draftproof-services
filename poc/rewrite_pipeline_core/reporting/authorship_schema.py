"""Authorship schema enrichment for older scan JSON consumed by rewrite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class AuthorshipSchemaEnrichmentDeps:
    calibrate_topk_risk: Callable[..., dict]
    split_sentences: Callable[[str], list[str]]
    build_layer3_input_from_text: Callable[..., Any]
    metric_decimal: Callable[[Any], float]
    layer3_scorer_factory: Callable[[], Any]


def enrich_report_authorship_schema(report_dict: dict, *, deps: AuthorshipSchemaEnrichmentDeps) -> dict:
    """Backfill current authorship fields for older saved scan JSON.

    Rewrite jobs often consume the saved scan JSON as their contract. If the
    scan was created before a detector/schema upgrade, the rewrite report can
    otherwise omit newer fields such as qualifying_text_ai_density even though
    the saved JSON still contains enough input text and scanner components to
    derive them cheaply.
    """
    if not isinstance(report_dict, dict):
        return report_dict

    badge = report_dict.get("ai_risk_badge") or {}
    if not isinstance(badge, dict):
        return report_dict

    ai_components = badge.get("ai_components") or {}
    has_density = isinstance(ai_components, dict) and "qualifying_text_ai_density" in ai_components
    if badge.get("authorship_rating") and has_density and "topk_calibrated_risk" in ai_components:
        return report_dict

    text = report_dict.get("input_text") or report_dict.get("original_text") or ""
    if not isinstance(text, str) or len(text.split()) < 300:
        return report_dict

    writing_components = badge.get("writing_components") or {}
    topk_calibration = deps.calibrate_topk_risk(
        ai_components.get("topk_pattern_raw", ai_components.get("topk_pattern")),
        eligible_sentence_count=max(3, len(deps.split_sentences(text))),
    )
    layer3_input = deps.build_layer3_input_from_text(
        text,
        predictability=deps.metric_decimal(ai_components.get("predictability")),
        topk_pattern=deps.metric_decimal(topk_calibration.get("topk_calibrated_risk")),
        generic_phrase_density=deps.metric_decimal(ai_components.get("generic_phrase_density")),
        broad_claim_risk=deps.metric_decimal(writing_components.get("broad_claim_risk")),
        citation_weakness_risk=deps.metric_decimal(writing_components.get("citation_weakness_risk")),
        unsupported_claim_risk=deps.metric_decimal(writing_components.get("unsupported_claim_risk")),
        source_grounding_strength=deps.metric_decimal(writing_components.get("source_grounding_strength")),
        domain_grounding_strength=deps.metric_decimal(writing_components.get("domain_grounding_strength")),
    )
    layer3 = deps.layer3_scorer_factory().score(layer3_input)
    enriched_ai_components = {k: round(v * 100, 2) for k, v in layer3.ai_phase.components.items()}
    enriched_ai_components["topk_authorship_component"] = enriched_ai_components.get("topk_pattern")
    enriched_ai_components.update(topk_calibration)
    enriched_ai_components["topk_pattern"] = topk_calibration.get(
        "topk_pattern_raw",
        ai_components.get("topk_pattern"),
    )

    enriched = dict(report_dict)
    enriched_badge = dict(badge)
    enriched_badge.update({
        "tier": layer3.tier.value,
        "ai_likelihood_score": round(layer3.ai_likelihood_score * 100, 2),
        "authorship_rating": layer3.authorship_rating,
        "authorship_rating_label": layer3.authorship_rating.get("label"),
        "authorship_rating_code": layer3.authorship_rating.get("code"),
        "ai_cluster_boost": round(layer3.ai_cluster_boost * 100, 2) if layer3.ai_cluster_boost else 0,
        "ai_cluster_name": layer3.ai_cluster_name,
        "ai_components": enriched_ai_components,
        "writing_quality_tier": layer3.writing_quality_tier.value,
        "writing_quality_score": round(layer3.writing_quality_score * 100, 2),
        "writing_components": {k: round(v * 100, 2) for k, v in layer3.writing_phase.components.items()},
        "review_priority": layer3.review_priority,
        "confidence": layer3.confidence.value,
        "reasons": layer3.reasons,
        "guardrails": layer3.guardrails,
        "schema_enriched_from_input_text": True,
    })
    enriched["ai_risk_badge"] = enriched_badge
    return enriched
