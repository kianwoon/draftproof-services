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
    semantic_uniformity_risk: float = 0.0
    discourse_regularity_risk: float = 0.0
    paraphrase_transformation_risk: float = 0.0
    signal_agreement_score: float = 0.0
    calibration_confidence: float = 0.0
    calibrated_ai_risk: float = 0.0
    reporting_suppression: float = 0.0


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

TRANSFORMATION_SIGNAL_METADATA: dict[str, dict[str, str]] = {
    "ai_likelihood": {
        "label": "AI likelihood",
        "description": "Statistical AI-style signal based on predictability, token concentration, generic phrasing, and sentence regularity.",
        "family": "statistical",
        "higher_score_means": "Text shape is more consistent with Transformer-style generation.",
    },
    "adjusted_ai_risk": {
        "label": "Adjusted AI risk",
        "description": "AI likelihood after reducing certainty for strong human anchoring and calibration checks.",
        "family": "calibration",
        "higher_score_means": "AI-style evidence remains visible after human-anchor suppression.",
    },
    "calibrated_ai_risk": {
        "label": "Calibrated AI risk",
        "description": "Final calibrated authorship risk after thresholding, signal agreement, and reporting safeguards.",
        "family": "calibration",
        "higher_score_means": "The report has stronger defensible AI-risk evidence after conservative calibration.",
    },
    "human_anchor_score": {
        "label": "Human anchor",
        "description": "Strength of concrete lived experience, local context, operational memory, and specific human reasoning.",
        "family": "human_anchor",
        "higher_score_means": "The writing contains stronger human-context evidence that should reduce AI certainty.",
    },
    "human_anchor_discount": {
        "label": "Human anchor discount",
        "description": "How much strong human anchoring reduces AI certainty in the calibrated risk model.",
        "family": "human_anchor",
        "higher_score_means": "Human-anchor evidence is suppressing more of the raw AI likelihood.",
    },
    "rewrite_smoothness": {
        "label": "Rewrite smoothness",
        "description": "Likelihood that a human draft was polished into cleaner, more even AI-assisted prose.",
        "family": "rewrite",
        "higher_score_means": "The writing looks more uniformly polished than the underlying ideas or evidence.",
    },
    "semantic_uniformity_risk": {
        "label": "Semantic uniformity",
        "description": "Embedding-based signal for overly even paragraph meaning or low semantic shape variation.",
        "family": "semantic_embedding",
        "higher_score_means": "Paragraph meanings are unusually uniform or mechanically similar.",
    },
    "discourse_regularity_risk": {
        "label": "Discourse regularity",
        "description": "Embedding-based signal for unusually regular paragraph progression and smooth discourse flow.",
        "family": "semantic_embedding",
        "higher_score_means": "Idea progression is unusually smooth or regular across adjacent paragraphs.",
    },
    "source_similarity": {
        "label": "Source similarity",
        "description": "Meaning-level closeness to source material, useful for detecting paraphrased source content.",
        "family": "source_similarity",
        "higher_score_means": "Submitted meaning remains close to a known source.",
    },
    "surface_similarity": {
        "label": "Surface similarity",
        "description": "Wording-level closeness to source material after direct text comparison.",
        "family": "source_similarity",
        "higher_score_means": "Submitted wording remains close to a known source.",
    },
    "paraphrase_transformation_risk": {
        "label": "Paraphrase transformation",
        "description": "Risk that source meaning was retained while wording and sentence structure were heavily changed.",
        "family": "paraphrase",
        "higher_score_means": "The source meaning appears preserved while surface wording diverges.",
    },
    "outline_to_text_expansion": {
        "label": "Expansion pattern",
        "description": "Risk that short notes or an outline were expanded into longer prose with limited new evidence.",
        "family": "expansion",
        "higher_score_means": "The document shows more signs of outline-to-essay expansion.",
    },
    "section_style_variance": {
        "label": "Patchwork variance",
        "description": "Section-level style shifts that can suggest stitched writing from different chunks or passes.",
        "family": "patchwork",
        "higher_score_means": "Sections differ more strongly in style, density, or vocabulary.",
    },
    "citation_grounding_risk": {
        "label": "Grounding risk",
        "description": "Claims, citations, or academic statements that appear weakly supported by evidence.",
        "family": "grounding",
        "higher_score_means": "More claims need source checking or stronger evidence.",
    },
    "signal_agreement_score": {
        "label": "Signal agreement",
        "description": "How strongly separate scanner layers agree with each other instead of firing in isolation.",
        "family": "calibration",
        "higher_score_means": "More independent scanner layers point in the same direction.",
    },
    "calibration_confidence": {
        "label": "Calibration confidence",
        "description": "Confidence that available signals are strong enough and sufficiently aligned for reporting.",
        "family": "calibration",
        "higher_score_means": "The system has more confidence in the calibrated scan interpretation.",
    },
    "reporting_suppression": {
        "label": "Reporting suppression",
        "description": "Amount of risk held back because the evidence is uncertain, limited, or not institutionally defensible.",
        "family": "calibration",
        "higher_score_means": "More raw risk was suppressed before presenting the report.",
    },
}


def transformation_signal_metadata(key: str) -> dict[str, str]:
    """Return stable reader/mitigation metadata for a transformation signal."""
    meta = TRANSFORMATION_SIGNAL_METADATA.get(key, {})
    return {
        "label": meta.get("label") or key.replace("_", " "),
        "description": meta.get("description") or "Scanner signal used to interpret the transformation pattern.",
        "family": meta.get("family") or "scanner",
        "higher_score_means": meta.get("higher_score_means") or "The signal is stronger in the scan profile.",
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

    semantic_uniformity_risk = clamp(
        0.40 * clamp(layer3_input.paragraph_topic_uniformity_risk)
        + 0.25 * clamp(layer3_input.generic_assertion_risk)
        + 0.20 * clamp(layer3_input.qualifying_text_ai_density)
        + 0.15 * clamp(layer3_input.intra_paragraph_parallelism_risk)
    )

    discourse_regularity_risk = clamp(
        0.30 * clamp(layer3_input.paragraph_progression_risk)
        + 0.25 * clamp(layer3_input.paragraph_uniformity_risk)
        + 0.18 * clamp(layer3_input.signpost_paragraph_risk)
        + 0.15 * clamp(layer3_input.repeated_starter_risk)
        + 0.12 * clamp(layer3_input.formulaic_conclusion_risk)
    )

    paraphrase_transformation_risk = clamp(
        source_similarity
        * (1.0 - surface_similarity)
        * (0.65 + 0.35 * rewrite_smoothness)
    )

    raw_ai_likelihood = clamp(layer3_result.ai_likelihood_score)
    effective_human_anchor = human_anchor_score
    human_anchor_discount = clamp(effective_human_anchor * 0.45)
    adjusted_ai_risk = clamp(raw_ai_likelihood * (1.0 - human_anchor_discount))
    signal_agreement_score = _signal_agreement(
        adjusted_ai_risk,
        rewrite_smoothness,
        outline_to_text_expansion,
        semantic_uniformity_risk,
        discourse_regularity_risk,
        paraphrase_transformation_risk,
    )
    calibration_confidence = _calibration_confidence(
        word_count=layer3_input.word_count,
        sentence_count=layer3_input.sentence_count,
        signal_agreement_score=signal_agreement_score,
        human_anchor_discount=human_anchor_discount,
    )
    calibrated_ai_risk = clamp(
        adjusted_ai_risk
        * (0.72 + 0.28 * signal_agreement_score)
        * (0.70 + 0.30 * calibration_confidence)
    )
    reporting_suppression = clamp(1.0 - calibration_confidence)

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
        semantic_uniformity_risk=round(semantic_uniformity_risk, 4),
        discourse_regularity_risk=round(discourse_regularity_risk, 4),
        paraphrase_transformation_risk=round(paraphrase_transformation_risk, 4),
        signal_agreement_score=round(signal_agreement_score, 4),
        calibration_confidence=round(calibration_confidence, 4),
        calibrated_ai_risk=round(calibrated_ai_risk, 4),
        reporting_suppression=round(reporting_suppression, 4),
    )


def classify_transformation(features: TransformationFeatures) -> TransformationClassification:
    """Classify the likely writing transformation pattern.

    The order intentionally mirrors the practical taxonomy: strong provenance
    patterns first, then softer pattern explanations, then uncertain.
    """
    f = features
    evidence: list[str] = []
    effective_ai_risk = _adjusted_ai_risk(f)
    calibrated_ai_risk = _calibrated_ai_risk(f)

    if f.human_anchor_score < 0.20 and effective_ai_risk > 0.72 and calibrated_ai_risk >= 0.55:
        code = "fully_ai_written"
        evidence = ["very low human anchor", "high calibrated AI risk"]
    elif f.human_anchor_score >= 0.50 and f.rewrite_smoothness > 0.70:
        code = "ai_cleaned_human_writing"
        evidence = ["clear human anchor", "very smooth rewrite surface"]
    elif (
        f.paraphrase_transformation_risk > 0.45
        or (f.source_similarity > 0.60 and f.surface_similarity < 0.30)
    ):
        code = "ai_paraphrased"
        evidence = ["high meaning overlap with source", "low surface overlap"]
    elif f.outline_to_text_expansion > 0.70 and f.discourse_regularity_risk >= 0.35:
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
    if f.semantic_uniformity_risk >= 0.45:
        evidence.append("semantic uniformity signal")
    if f.calibration_confidence and f.calibration_confidence < 0.45:
        evidence.append("institutional reporting threshold suppressed")

    confidence = _confidence_for(code, f)
    serialized_features = {k: round(float(v), 4) for k, v in asdict(f).items()}
    serialized_features["adjusted_ai_risk"] = round(_adjusted_ai_risk(f), 4)
    serialized_features["human_anchor_discount"] = round(_human_anchor_discount(f), 4)
    serialized_features["calibrated_ai_risk"] = round(_calibrated_ai_risk(f), 4)
    serialized_features["reporting_suppression"] = round(_reporting_suppression(f), 4)
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
    return clamp(features.human_anchor_score * 0.45)


def _adjusted_ai_risk(features: TransformationFeatures) -> float:
    if features.adjusted_ai_risk > 0:
        return clamp(features.adjusted_ai_risk)
    return clamp(features.ai_likelihood * (1.0 - _human_anchor_discount(features)))


def _signal_agreement(*signals: float) -> float:
    values = [clamp(signal) for signal in signals]
    if not values:
        return 0.0
    active = sum(1 for value in values if value >= 0.35)
    strong = sum(1 for value in values if value >= 0.60)
    return clamp((active / len(values)) * 0.70 + (strong / len(values)) * 0.30)


def _coverage_score(word_count: int, sentence_count: int) -> float:
    if word_count >= 700 and sentence_count >= 25:
        return 0.95
    if word_count >= 350 and sentence_count >= 12:
        return 0.75
    if word_count >= 150 and sentence_count >= 6:
        return 0.55
    return 0.30


def _calibration_confidence(
    *,
    word_count: int,
    sentence_count: int,
    signal_agreement_score: float,
    human_anchor_discount: float,
) -> float:
    coverage = _coverage_score(word_count, sentence_count)
    confidence = (
        0.50 * coverage
        + 0.35 * clamp(signal_agreement_score)
        + 0.15 * (1.0 - clamp(human_anchor_discount))
    )
    return clamp(confidence)


def _calibrated_ai_risk(features: TransformationFeatures) -> float:
    if features.calibrated_ai_risk > 0:
        return clamp(features.calibrated_ai_risk)
    adjusted = _adjusted_ai_risk(features)
    agreement = features.signal_agreement_score or _signal_agreement(
        adjusted,
        features.rewrite_smoothness,
        features.outline_to_text_expansion,
        features.semantic_uniformity_risk,
        features.discourse_regularity_risk,
        features.paraphrase_transformation_risk,
    )
    confidence = features.calibration_confidence or 0.65
    return clamp(adjusted * (0.72 + 0.28 * agreement) * (0.70 + 0.30 * confidence))


def _reporting_suppression(features: TransformationFeatures) -> float:
    if features.reporting_suppression > 0:
        return clamp(features.reporting_suppression)
    if features.calibration_confidence > 0:
        return clamp(1.0 - features.calibration_confidence)
    return 0.0


def _confidence_for(code: str, features: TransformationFeatures) -> str:
    if code == "human_uncertain":
        return "low"

    values = asdict(features)
    values["adjusted_ai_risk"] = _adjusted_ai_risk(features)
    values["calibrated_ai_risk"] = _calibrated_ai_risk(features)
    strongest = max(float(v) for v in values.values())
    if features.calibration_confidence and features.calibration_confidence < 0.45:
        return "low"
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
