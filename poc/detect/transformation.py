"""Writing transformation classifier.

This layer classifies the *pattern of transformation* behind a scan result.
It deliberately sits after the mechanical AI likelihood and writing-quality
scorers so the product can explain what kind of risk is present instead of
reducing every case to "AI or not AI".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

from .layer3_scoring import Layer3Input, Layer3Result, clamp


@dataclass(frozen=True)
class TransformationFeatures:
    ai_likelihood: float
    human_anchor_score: float
    rewrite_smoothness: float
    source_similarity: float
    surface_similarity: float
    outline_to_text_expansion: float
    section_style_variance: float
    citation_grounding_risk: float
    adjusted_ai_risk: float = 0.0
    human_anchor_discount: float = 0.0


@dataclass(frozen=True)
class TransformationClassification:
    code: str
    label: str
    confidence: str
    evidence: list[str]
    features: dict[str, float]
    is_verdict: bool = False


_LABELS = {
    "fully_ai_written": "Fully AI-written pattern",
    "ai_cleaned_human_writing": "AI-cleaned human writing pattern",
    "ai_paraphrased": "AI-paraphrased source pattern",
    "ai_expanded": "AI-expanded outline pattern",
    "ai_stitched_patchwork": "AI-stitched / patchwork pattern",
    "ai_cited_weakly_grounded": "AI-cited but weakly grounded pattern",
    "human_uncertain": "Human / uncertain pattern",
}


def _similarity_features(similarity_summary: Optional[Any]) -> tuple[float, float]:
    """Return semantic/source similarity and lexical/surface similarity."""
    if not similarity_summary:
        return 0.0, 0.0

    matches = getattr(similarity_summary, "matches", None) or []
    source_similarity = 0.0
    surface_similarity = 0.0
    for match in matches:
        if not isinstance(match, dict):
            continue
        semantic = clamp(match.get("semantic_score"))
        lexical = max(
            clamp(match.get("exact_score")),
            clamp(match.get("fuzzy_score")),
        )
        source_similarity = max(source_similarity, semantic)
        surface_similarity = max(surface_similarity, lexical)

    return source_similarity, surface_similarity


def build_transformation_features(
    layer3_input: Layer3Input,
    layer3_result: Layer3Result,
    *,
    similarity_summary: Optional[Any] = None,
) -> TransformationFeatures:
    """Normalize existing scanner outputs into transformation-pattern features."""
    source_similarity, surface_similarity = _similarity_features(similarity_summary)

    lived_anchor = 1.0 - clamp(layer3_input.lived_detail_risk)
    source_anchor = clamp(layer3_input.source_grounding_strength)
    domain_anchor = clamp(layer3_input.domain_grounding_strength)
    human_anchor_score = clamp(
        0.55 * lived_anchor + 0.25 * source_anchor + 0.20 * domain_anchor
    )

    rewrite_smoothness = clamp(
        0.35 * clamp(layer3_input.predictability)
        + 0.25 * clamp(layer3_input.topk_pattern)
        + 0.15 * (1.0 - clamp(layer3_input.burstiness_risk))
        + 0.15 * clamp(layer3_input.repeated_sentence_structure_risk)
        + 0.10 * clamp(layer3_input.balanced_hedging_risk)
    )

    outline_to_text_expansion = clamp(
        0.45 * clamp(layer3_input.qualifying_text_ai_density)
        + 0.25 * clamp(layer3_input.signpost_paragraph_risk)
        + 0.20 * clamp(layer3_input.paragraph_progression_risk)
        + 0.10 * clamp(layer3_input.generic_assertion_risk)
    )

    section_style_variance = max(
        clamp(layer3_input.style_shift_risk),
        clamp(layer3_input.draft_evolution_jump_risk),
        clamp(layer3_input.paragraph_topic_uniformity_risk),
    )

    citation_grounding_risk = max(
        clamp(layer3_input.citation_weakness_risk),
        clamp(layer3_input.unsupported_claim_risk),
        1.0 - clamp(layer3_input.source_grounding_strength),
    )

    raw_ai_likelihood = clamp(layer3_result.ai_likelihood_score)
    effective_human_anchor = clamp(
        human_anchor_score * (1.0 - 0.35 * citation_grounding_risk)
    )
    human_anchor_discount = clamp(effective_human_anchor * 0.45)
    adjusted_ai_risk = clamp(raw_ai_likelihood * (1.0 - human_anchor_discount))

    return TransformationFeatures(
        ai_likelihood=raw_ai_likelihood,
        human_anchor_score=round(human_anchor_score, 4),
        rewrite_smoothness=round(rewrite_smoothness, 4),
        source_similarity=round(source_similarity, 4),
        surface_similarity=round(surface_similarity, 4),
        outline_to_text_expansion=round(outline_to_text_expansion, 4),
        section_style_variance=round(section_style_variance, 4),
        citation_grounding_risk=round(citation_grounding_risk, 4),
        adjusted_ai_risk=round(adjusted_ai_risk, 4),
        human_anchor_discount=round(human_anchor_discount, 4),
    )


def classify_transformation(features: TransformationFeatures) -> TransformationClassification:
    """Classify the likely writing transformation pattern.

    The order intentionally mirrors the practical taxonomy: strong provenance
    patterns first, then softer pattern explanations, then uncertain.
    """
    f = features
    evidence: list[str] = []
    effective_ai_risk = _adjusted_ai_risk(f)

    if f.human_anchor_score < 0.20 and effective_ai_risk > 0.72:
        code = "fully_ai_written"
        evidence = ["very low human anchor", "high adjusted AI risk"]
    elif f.human_anchor_score >= 0.50 and f.rewrite_smoothness > 0.70:
        code = "ai_cleaned_human_writing"
        evidence = ["clear human anchor", "very smooth rewrite surface"]
    elif f.source_similarity > 0.60 and f.surface_similarity < 0.30:
        code = "ai_paraphrased"
        evidence = ["high meaning overlap with source", "low surface overlap"]
    elif f.outline_to_text_expansion > 0.70:
        code = "ai_expanded"
        evidence = ["outline-to-text expansion pattern", "thin new grounding"]
    elif f.section_style_variance > 0.60:
        code = "ai_stitched_patchwork"
        evidence = ["section-level style variance"]
    elif (
        f.citation_grounding_risk > 0.60
        and effective_ai_risk >= 0.30
        and f.human_anchor_score < 0.50
        and (
            f.rewrite_smoothness >= 0.35
            or f.outline_to_text_expansion >= 0.35
            or f.source_similarity >= 0.35
        )
    ):
        code = "ai_cited_weakly_grounded"
        evidence = ["weak source grounding", "AI-style signals present"]
    else:
        code = "human_uncertain"
        evidence = ["no single transformation pattern dominates"]

    if _human_anchor_discount(f) >= 0.15:
        evidence.append("human anchor reduced AI certainty")

    confidence = _confidence_for(code, f)
    serialized_features = {k: round(float(v), 4) for k, v in asdict(f).items()}
    serialized_features["adjusted_ai_risk"] = round(_adjusted_ai_risk(f), 4)
    serialized_features["human_anchor_discount"] = round(_human_anchor_discount(f), 4)
    return TransformationClassification(
        code=code,
        label=_LABELS[code],
        confidence=confidence,
        evidence=evidence,
        features=serialized_features,
    )


def _human_anchor_discount(features: TransformationFeatures) -> float:
    if features.human_anchor_discount > 0:
        return clamp(features.human_anchor_discount)
    effective_human_anchor = clamp(
        features.human_anchor_score * (1.0 - 0.35 * features.citation_grounding_risk)
    )
    return clamp(effective_human_anchor * 0.45)


def _adjusted_ai_risk(features: TransformationFeatures) -> float:
    if features.adjusted_ai_risk > 0:
        return clamp(features.adjusted_ai_risk)
    return clamp(features.ai_likelihood * (1.0 - _human_anchor_discount(features)))


def _confidence_for(code: str, features: TransformationFeatures) -> str:
    if code == "human_uncertain":
        return "low"

    values = asdict(features)
    values["adjusted_ai_risk"] = _adjusted_ai_risk(features)
    strongest = max(float(v) for v in values.values())
    if strongest >= 0.75:
        return "high"
    if strongest >= 0.55:
        return "medium"
    return "low"


def classify_transformation_from_scan(
    layer3_input: Layer3Input,
    layer3_result: Layer3Result,
    *,
    similarity_summary: Optional[Any] = None,
) -> TransformationClassification:
    features = build_transformation_features(
        layer3_input,
        layer3_result,
        similarity_summary=similarity_summary,
    )
    return classify_transformation(features)
