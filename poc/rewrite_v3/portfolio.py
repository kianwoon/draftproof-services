"""Portfolio selector for rewrite V3 candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

from .calibration_store import records_for_family
from .candidate_features import CandidateFeatures, features_from_trace


@dataclass(frozen=True)
class PortfolioScore:
    candidate_index: int
    score: float
    features: CandidateFeatures
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = self.features.to_dict()
        return payload


def _calibration_adjustment(family: str, features: CandidateFeatures) -> tuple[float, list[str]]:
    adjustment = 0.0
    reasons: list[str] = []
    records = [record for record in records_for_family(family) if record.features is not None]
    if not records:
        return adjustment, reasons
    for record in records:
        assert record.features is not None
        distance = (
            abs(features.topk_delta - record.features.topk_delta) / 30.0
            + abs(features.wq_delta - record.features.wq_delta) / 30.0
            + abs(features.ai_delta - record.features.ai_delta) / 50.0
            + abs(features.compression_ratio - record.features.compression_ratio)
            + abs(features.proxy_reason_count - record.features.proxy_reason_count) / 4.0
        )
        similarity = max(0.0, 1.0 - min(distance, 1.0))
        if similarity <= 0:
            continue
        if record.passed_external:
            adjustment += similarity * 12.0
            reasons.append("near_positive_external_calibration")
        else:
            adjustment -= similarity * 16.0
            reasons.append("near_negative_external_calibration")
    return adjustment, reasons


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _material_ai_drop_for_review(features: CandidateFeatures) -> bool:
    threshold = _float_env("DRAFTPROOF_REWRITE_V3_MATERIAL_AI_DROP_FOR_REVIEW", 8.0)
    return (
        features.validation_passed
        and features.compression_accepted
        and features.semantic_safe
        and features.ai_delta >= threshold
    )


def score_candidate(item: dict[str, Any], *, family: str, index: int) -> PortfolioScore:
    trace = item.get("trace") if isinstance(item.get("trace"), dict) else {}
    features = features_from_trace(trace)
    score = 0.0
    reasons: list[str] = []
    outcome = str(trace.get("candidate_outcome") or "")
    material_ai_drop = _material_ai_drop_for_review(features)
    if outcome == "valid_detector_improved":
        score += 55.0
        reasons.append("valid_detector_improved")
    elif outcome == "invalid_detector_improved":
        score += 15.0
        reasons.append("invalid_detector_improved")
    elif outcome == "valid_no_detector_movement":
        if material_ai_drop:
            score += 20.0
            reasons.append("material_ai_drop_without_detector_gate")
        else:
            score -= 35.0
            reasons.append("no_detector_movement")
    elif outcome == "invalid_no_detector_movement":
        score -= 100.0
        reasons.append("no_detector_movement")
    elif outcome == "corrupted_output":
        score -= 220.0
        reasons.append("corrupted_output")
    elif outcome.startswith("generation_failed"):
        score -= 260.0
        reasons.append("generation_failed")
    if features.validation_passed:
        score += 40.0
        reasons.append("valid_structure_and_anchors")
    else:
        score -= 80.0
        reasons.append("invalid_structure_or_anchors")
    if features.compression_accepted:
        score += 20.0
        reasons.append("compression_accepted")
    else:
        score -= 40.0
        reasons.append("compression_rejected")
    if features.semantic_safe:
        score += 30.0
        reasons.append("semantic_safe")
    else:
        score -= 100.0
        reasons.append("semantic_unsafe")
    if not features.target_gate_passed:
        if material_ai_drop:
            score -= 35.0
            reasons.append("target_gate_failed_but_material_ai_drop")
        else:
            score -= 80.0
            reasons.append("target_gate_failed")
    if features.footprint_risk_drop < 0:
        score -= 160.0 + min(abs(features.footprint_risk_drop), 30.0) * 8.0
        reasons.append("footprint_regression")
    else:
        score += min(features.footprint_risk_drop, 25.0) * 3.0
    if features.target_risk_drop < 0:
        score -= 80.0 + min(abs(features.target_risk_drop), 20.0) * 4.0
        reasons.append("target_regression")
    else:
        score += min(features.target_risk_drop, 20.0) * 2.0
    score += min(max(features.topk_delta, -20.0), 35.0) * 2.0
    score += min(max(features.ai_delta, -20.0), 50.0) * 1.6
    score -= features.fraction_ai * 80.0
    score -= features.fraction_ai_assisted * 35.0
    score += features.fraction_human * 30.0
    if features.ownership_gate_active:
        if features.ownership_gate_passed:
            score += min(features.ownership_score, 15.0) * 4.0
            score += min(features.ownership_change_count, 4) * 8.0
            reasons.append("ownership_gate_passed")
        else:
            score -= 140.0
            reasons.append("ownership_gate_failed")
            if features.fraction_human <= 0.0:
                score -= 60.0
                reasons.append("zero_human_fraction_with_ownership_blocker")
    score -= min(features.max_ai_window_words, 300.0) * 0.12
    score -= max(features.wq_delta, 0.0) * 1.4
    score -= features.proxy_reason_count * 10.0
    calibration, calibration_reasons = _calibration_adjustment(family, features)
    score += calibration
    reasons.extend(calibration_reasons)
    return PortfolioScore(candidate_index=index, score=round(score, 3), features=features, reasons=tuple(reasons))


def select_portfolio_candidate(candidate_evaluations: list[dict[str, Any]], *, family: str) -> tuple[int, list[dict[str, Any]]]:
    scores = [
        score_candidate(item, family=family, index=index)
        for index, item in enumerate(candidate_evaluations)
    ]
    best = max(scores, key=lambda item: item.score)
    return best.candidate_index, [score.to_dict() for score in scores]
