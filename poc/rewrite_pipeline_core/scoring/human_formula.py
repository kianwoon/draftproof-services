from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class HumanFormulaDriverDeps:
    env_flag: Callable[[str, bool], bool]
    float_env: Callable[[str, float], float]
    contribution_scores: Callable[[dict | None], dict]


def human_formula_driver_status(
    original_report: dict | None,
    candidate_report: dict | None,
    deps: HumanFormulaDriverDeps,
) -> dict:
    """Track the actual transformation drivers behind Human Contribution."""
    def features(report: dict | None) -> dict:
        if not isinstance(report, dict):
            return {}
        badge = report.get("ai_risk_badge") or {}
        transform = badge.get("transformation_classification") or {}
        return transform.get("features") or {}

    def fnum(mapping: dict, key: str) -> float:
        value = mapping.get(key)
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def raw_parts(mapping: dict) -> dict:
        max_similarity = max(
            fnum(mapping, "source_similarity"),
            fnum(mapping, "surface_similarity"),
        )
        human_raw = (
            fnum(mapping, "human_anchor_score") * 0.45
            + (1.0 - fnum(mapping, "rewrite_smoothness")) * 0.20
            + (1.0 - max_similarity) * 0.10
        )
        ai_raw = (
            fnum(mapping, "ai_likelihood") * 0.55
            + fnum(mapping, "rewrite_smoothness") * 0.25
            + fnum(mapping, "outline_to_text_expansion") * 0.15
            + fnum(mapping, "semantic_uniformity_risk") * 0.10
            + fnum(mapping, "discourse_regularity_risk") * 0.05
            + fnum(mapping, "section_style_variance") * 0.05
            + fnum(mapping, "source_similarity") * 0.05
        )
        return {"human_raw": human_raw, "ai_raw": ai_raw}

    original_features = features(original_report)
    candidate_features = features(candidate_report)
    original_parts = raw_parts(original_features)
    candidate_parts = raw_parts(candidate_features)
    driver_keys = [
        "ai_likelihood",
        "rewrite_smoothness",
        "outline_to_text_expansion",
        "semantic_uniformity_risk",
        "discourse_regularity_risk",
        "section_style_variance",
        "source_similarity",
    ]
    drops = {
        key: round(fnum(original_features, key) - fnum(candidate_features, key), 4)
        for key in driver_keys
    }
    regressions = {
        key: round(abs(value), 4)
        for key, value in drops.items()
        if value < 0
    }
    ai_raw_drop = original_parts["ai_raw"] - candidate_parts["ai_raw"]
    human_raw_gain = candidate_parts["human_raw"] - original_parts["human_raw"]
    contribution = deps.contribution_scores(candidate_report)
    original_contribution = deps.contribution_scores(original_report)
    human_delta = (
        float(contribution.get("human")) - float(original_contribution.get("human"))
        if isinstance(contribution.get("human"), (int, float))
        and isinstance(original_contribution.get("human"), (int, float))
        else 0.0
    )
    target_human = deps.float_env("DRAFTPROOF_AUTHENTICITY_TARGET_HUMAN", 80.0)
    required = bool(
        isinstance(original_contribution.get("human"), (int, float))
        and float(original_contribution.get("human")) < target_human
        and deps.env_flag("DRAFTPROOF_REQUIRE_HUMAN_FORMULA_DRIVER_PROGRESS", True)
    )
    min_ai_raw_drop = deps.float_env("DRAFTPROOF_HUMAN_FORMULA_MIN_AI_RAW_DROP", 0.04)
    min_human_gain = deps.float_env("DRAFTPROOF_HUMAN_FORMULA_MIN_HUMAN_GAIN", 4.0)
    max_driver_regression = deps.float_env("DRAFTPROOF_HUMAN_FORMULA_MAX_DRIVER_REGRESSION", 0.04)
    safe_min_ai_raw_drop = deps.float_env(
        "DRAFTPROOF_HUMAN_FORMULA_SAFE_MIN_AI_RAW_DROP",
        min_ai_raw_drop * 0.75,
    )
    safe_min_human_gain = deps.float_env(
        "DRAFTPROOF_HUMAN_FORMULA_SAFE_MIN_HUMAN_GAIN",
        1.0,
    )
    total_regression = sum(regressions.values())
    safe_partial_progress = bool(
        required
        and float(human_delta) >= safe_min_human_gain
        and ai_raw_drop >= safe_min_ai_raw_drop
        and human_raw_gain >= 0.0
        and total_regression <= max_driver_regression
    )
    clears = bool(
        not required
        or float(human_delta) >= min_human_gain
        or safe_partial_progress
        or (
            ai_raw_drop >= min_ai_raw_drop
            and human_raw_gain >= -0.01
            and total_regression <= max_driver_regression
        )
    )
    return {
        "required": required,
        "cleared": clears,
        "reason": "" if clears else "human_formula_drivers_not_reduced",
        "human_delta": round(human_delta, 3),
        "target_human": target_human,
        "ai_raw_drop": round(ai_raw_drop, 4),
        "human_raw_gain": round(human_raw_gain, 4),
        "safe_partial_progress": safe_partial_progress,
        "thresholds": {
            "min_ai_raw_drop": min_ai_raw_drop,
            "min_human_gain": min_human_gain,
            "safe_min_ai_raw_drop": safe_min_ai_raw_drop,
            "safe_min_human_gain": safe_min_human_gain,
            "max_driver_regression": max_driver_regression,
        },
        "driver_drops": drops,
        "driver_regressions": regressions,
        "total_driver_regression": round(total_regression, 4),
        "original_raw": {key: round(value, 4) for key, value in original_parts.items()},
        "candidate_raw": {key: round(value, 4) for key, value in candidate_parts.items()},
    }
