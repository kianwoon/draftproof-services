"""Central signal controller for Rewrite V2 experiments.

The controller expresses the rewrite objective as a weighted signal profile:

* Human / authenticity signals: 45%, higher is better.
* AI authorship signals: 35%, lower is better.
* Quality / calibration signals: 20%, must support trust.

It does not call an LLM. It turns scan output into a generation plan and scores
candidate scan output using the same contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


GROUP_WEIGHTS: dict[str, float] = {
    "human_authenticity": 0.45,
    "anti_ai_authorship": 0.35,
    "quality_calibration_trust": 0.20,
}


AI_AUTHORSHIP_SIGNAL_KEYS: tuple[str, ...] = (
    "raw_topk_predictability",
    "calibrated_topk_risk",
    "expansion_pattern",
    "rewrite_smoothness",
    "ai_likelihood",
    "patchwork_variance",
    "semantic_uniformity",
    "discourse_regularity",
)

HUMAN_AUTHENTICITY_SIGNAL_KEYS: tuple[str, ...] = (
    "human_anchor",
    "human_anchor_discount",
)

QUALITY_CALIBRATION_SIGNAL_KEYS: tuple[str, ...] = (
    "grounding_risk",
    "calibration_confidence",
    "reporting_suppression",
    "adjusted_ai_risk",
    "signal_agreement",
    "calibrated_ai_risk",
    "paraphrase_transformation",
    "source_similarity",
    "surface_similarity",
)


@dataclass(frozen=True)
class CentralSignalProfile:
    ai_authorship: dict[str, float]
    human_authenticity: dict[str, float]
    quality_calibration: dict[str, float]
    group_scores: dict[str, float]
    weighted_human_target_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextualJudgmentPlan:
    strategy_id: str
    generation_objective: dict[str, Any]
    constraints: dict[str, Any]
    content_operations: list[dict[str, Any]] = field(default_factory=list)
    candidate_selection_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number <= 1.0:
        number *= 100.0
    return max(0.0, min(100.0, number))


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _feature(report: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    report = report or {}
    badge = report.get("ai_risk_badge") or {}
    ai_components = badge.get("ai_components") or {}
    writing_components = badge.get("writing_components") or {}
    transform = badge.get("transformation_classification") or {}
    features = transform.get("features") or {}
    integrity_layers = ((report.get("integrity_layers") or {}).get("layers") or {})

    if key == "raw_topk_predictability":
        return _clamp_score(ai_components.get("topk_pattern_raw") or ai_components.get("topk_pattern"))
    if key == "calibrated_topk_risk":
        return _clamp_score(ai_components.get("topk_calibrated_risk"))
    if key == "expansion_pattern":
        return _clamp_score(features.get("outline_to_text_expansion"))
    if key == "rewrite_smoothness":
        return _clamp_score(features.get("rewrite_smoothness"))
    if key == "ai_likelihood":
        return _clamp_score(badge.get("ai_likelihood_score") or features.get("ai_likelihood"))
    if key == "patchwork_variance":
        return _clamp_score(features.get("section_style_variance"))
    if key == "semantic_uniformity":
        return _clamp_score(features.get("semantic_uniformity_risk"))
    if key == "discourse_regularity":
        return _clamp_score(features.get("discourse_regularity_risk"))

    if key == "human_anchor":
        layer = integrity_layers.get("human_contribution_signal") or {}
        layer_score = layer.get("score")
        if layer_score is not None:
            return _clamp_score(layer_score)
        return _clamp_score(features.get("human_anchor_score"))
    if key == "human_anchor_discount":
        return _clamp_score(features.get("human_anchor_discount"))

    if key == "grounding_risk":
        layer = integrity_layers.get("grounding_quality_risk") or {}
        return _clamp_score(layer.get("score") or writing_components.get("source_grounding_risk"))
    if key == "calibration_confidence":
        return _clamp_score(features.get("calibration_confidence"))
    if key == "reporting_suppression":
        return _clamp_score(features.get("reporting_suppression"))
    if key == "adjusted_ai_risk":
        return _clamp_score(features.get("adjusted_ai_risk"))
    if key == "signal_agreement":
        return _clamp_score(features.get("signal_agreement_score"))
    if key == "calibrated_ai_risk":
        return _clamp_score(features.get("calibrated_ai_risk"))
    if key == "paraphrase_transformation":
        return _clamp_score(features.get("paraphrase_transformation_risk"))
    if key == "source_similarity":
        return _clamp_score(features.get("source_similarity"))
    if key == "surface_similarity":
        return _clamp_score(features.get("surface_similarity"))
    return default


def extract_central_signal_profile(report: dict[str, Any] | None) -> CentralSignalProfile:
    """Extract the 19-signal central profile from a DraftProof scan report."""

    ai = {key: _feature(report, key) for key in AI_AUTHORSHIP_SIGNAL_KEYS}
    human = {key: _feature(report, key) for key in HUMAN_AUTHENTICITY_SIGNAL_KEYS}
    quality = {key: _feature(report, key) for key in QUALITY_CALIBRATION_SIGNAL_KEYS}

    ai_risk = _avg(list(ai.values()))
    anti_ai = round(100.0 - ai_risk, 2)
    human_strength = _avg(list(human.values()))

    quality_trust_inputs = [
        100.0 - quality["grounding_risk"],
        quality["calibration_confidence"],
        quality["reporting_suppression"],
        100.0 - quality["adjusted_ai_risk"],
        100.0 - quality["calibrated_ai_risk"],
        100.0 - quality["paraphrase_transformation"],
        quality["source_similarity"],
        quality["surface_similarity"],
        100.0 - quality["signal_agreement"],
    ]
    quality_trust = _avg(quality_trust_inputs)

    weighted = round(
        GROUP_WEIGHTS["human_authenticity"] * human_strength
        + GROUP_WEIGHTS["anti_ai_authorship"] * anti_ai
        + GROUP_WEIGHTS["quality_calibration_trust"] * quality_trust,
        2,
    )
    return CentralSignalProfile(
        ai_authorship=ai,
        human_authenticity=human,
        quality_calibration=quality,
        group_scores={
            "ai_authorship_risk": ai_risk,
            "anti_ai_authorship": anti_ai,
            "human_authenticity": human_strength,
            "quality_calibration_trust": quality_trust,
        },
        weighted_human_target_score=weighted,
    )


def _content_units_for_paragraph(text: str) -> list[dict[str, Any]]:
    """Build coarse content units without regex or keyword matching."""

    sentences = [part.strip() for part in text.replace("?", ".").replace("!", ".").split(".") if part.strip()]
    units: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences, start=1):
        units.append(
            {
                "unit_id": f"u{index}",
                "source_sentence_index": index,
                "meaning": sentence,
            }
        )
    return units


def build_contextual_judgment_plan(
    *,
    source_text: str,
    scan_report: dict[str, Any] | None,
    external_target_pattern: str | None = None,
) -> ContextualJudgmentPlan:
    """Build the central module plan for contextual judgment reconstruction."""

    profile = extract_central_signal_profile(scan_report)
    words = len(source_text.split())
    min_words = max(1, int(words * 0.85))
    sentence_count = max(1, len(_content_units_for_paragraph(source_text)))

    drivers = []
    ai = profile.ai_authorship
    quality = profile.quality_calibration
    if ai["raw_topk_predictability"] >= 70.0 or ai["calibrated_topk_risk"] >= 55.0:
        drivers.append("route_variation")
    if quality["grounding_risk"] >= 60.0:
        drivers.append("safe_contextual_anchor_expansion")
    if ai["rewrite_smoothness"] >= 40.0 or ai["semantic_uniformity"] >= 30.0:
        drivers.append("reasoning_interruption")
    if not drivers:
        drivers.append("minimal_contextual_judgment")

    operations = [
        {
            "operation": "safe_contextual_anchor_expansion",
            "instruction": "Turn broad categories into safe contextual anchors already implied by the source.",
        },
        {
            "operation": "reasoning_interruption",
            "instruction": "Add one plain reasoning turn that prevents a clean list-to-summary route.",
        },
        {
            "operation": "non_formulaic_judgment",
            "instruction": "End with a relevant judgment that avoids a balanced both-sides formula.",
        },
    ]

    return ContextualJudgmentPlan(
        strategy_id="central_contextual_judgment_v1",
        generation_objective={
            "weights": GROUP_WEIGHTS,
            "current_profile": profile.to_dict(),
            "target": "high authenticity, low AI authorship risk, and trustworthy quality calibration",
            "external_target_pattern": external_target_pattern or "contextual anchors plus imperfect reasoning turn",
            "primary_drivers": drivers,
        },
        constraints={
            "source_word_count": words,
            "minimum_words": min_words,
            "preferred_sentence_count": sentence_count,
            "preserve_content_units": _content_units_for_paragraph(source_text),
            "do_not_add_unsupported_facts": True,
            "do_not_use_detector_language_in_prompt": True,
        },
        content_operations=operations,
        candidate_selection_policy={
            "score_with_central_profile": True,
            "reject_if_word_count_below_minimum": True,
            "reject_if_content_unit_missing": True,
            "prefer_external_calibration_when_available": True,
            "strict_success_requires_goal_contract": True,
        },
    )


def score_candidate_against_central_profile(
    candidate_report: dict[str, Any] | None,
    *,
    baseline_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a candidate with the central 45/35/20 contract."""

    candidate = extract_central_signal_profile(candidate_report)
    baseline = extract_central_signal_profile(baseline_report) if baseline_report else None
    payload: dict[str, Any] = {
        "candidate_profile": candidate.to_dict(),
        "weighted_human_target_score": candidate.weighted_human_target_score,
    }
    if baseline:
        payload["baseline_weighted_human_target_score"] = baseline.weighted_human_target_score
        payload["weighted_delta"] = round(
            candidate.weighted_human_target_score - baseline.weighted_human_target_score,
            2,
        )
        payload["group_deltas"] = {
            key: round(
                candidate.group_scores.get(key, 0.0) - baseline.group_scores.get(key, 0.0),
                2,
            )
            for key in candidate.group_scores
        }
    return payload
