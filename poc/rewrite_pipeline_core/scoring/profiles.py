"""Score/profile helpers for the rewrite pipeline facade.

These helpers are intentionally read-only over report dictionaries. They keep
formula, footprint, and candidate-ranking calculations out of the pipeline
orchestrator without changing the public private-helper imports used by tests.
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher

from detect.topk_calibration import calibrate_topk_risk
from detect.turnitin_like import (
    TURNITIN_LIKE_COMPONENT_WEIGHTS,
    TURNITIN_LIKE_TARGET_AI_SCORE,
    turnitin_like_ai_profile_from_report,
)
from poc.report.contribution import contribution_pair


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _transformation_features(report_dict: dict | None) -> dict:
    if not isinstance(report_dict, dict):
        return {}
    badge = report_dict.get("ai_risk_badge") or {}
    transform = badge.get("transformation_classification") or {}
    features = transform.get("features")
    return features if isinstance(features, dict) else {}


def _feature_percent(report_dict: dict | None, key: str):
    value = _transformation_features(report_dict).get(key)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.0 else value


def _contribution_score_dict(human, ai_transformation) -> dict:
    human_score, ai_score = contribution_pair(human, ai_transformation)
    if human_score is None and ai_score is None:
        return {"human": None, "ai_transformation": None}
    return {
        "human": round(float(human_score or 0.0), 3),
        "ai_transformation": round(float(ai_score or 0.0), 3),
    }


def _contribution_scores(report_dict: dict | None) -> dict:
    """Extract the Human Contribution / AI Transformation product scores."""
    if not isinstance(report_dict, dict):
        return {"human": None, "ai_transformation": None}
    integrity = (
        report_dict.get("integrity_layers")
        or ((report_dict.get("scan_intelligence") or {}).get("integrity_layers") or {})
    )
    layers = integrity.get("layers") if isinstance(integrity, dict) else {}
    if isinstance(layers, dict):
        human_layer = layers.get("human_contribution_signal") or {}
        transform_layer = layers.get("ai_transformation_risk") or {}
        human_score = human_layer.get("score")
        transform_score = transform_layer.get("score")
        if isinstance(human_score, (int, float)) or isinstance(transform_score, (int, float)):
            return _contribution_score_dict(human_score, transform_score)
    contribution = (
        ((report_dict.get("scan_intelligence") or {}).get("transformation") or {})
        .get("contribution")
        or {}
    )

    def _score(*keys):
        for key in keys:
            value = contribution.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    human = _score("human_contribution_ratio", "human_contribution", "human_ratio")
    ai_transformation = _score(
        "ai_transformation_ratio",
        "ai_transformation",
        "transformation_ratio",
    )
    return _contribution_score_dict(human, ai_transformation)


def _integrity_scores(report_dict: dict | None) -> dict:
    if not isinstance(report_dict, dict):
        return {"ai_authorship": None, "grounding": None}
    integrity = (
        report_dict.get("integrity_layers")
        or ((report_dict.get("scan_intelligence") or {}).get("integrity_layers") or {})
    )
    layers = integrity.get("layers") if isinstance(integrity, dict) else {}
    if not isinstance(layers, dict):
        return {"ai_authorship": None, "grounding": None}

    def _score(layer_name: str):
        value = (layers.get(layer_name) or {}).get("score")
        return float(value) if isinstance(value, (int, float)) else None

    return {
        "ai_authorship": _score("ai_authorship_risk"),
        "ai_transformation": _score("ai_transformation_risk"),
        "grounding": _score("grounding_quality_risk"),
        "human": _score("human_contribution_signal"),
    }


def _blocker_scores(report_dict: dict | None) -> dict:
    if not isinstance(report_dict, dict):
        return {}
    badge = report_dict.get("ai_risk_badge") or {}
    ai = badge.get("ai_components") or {}
    writing = badge.get("writing_components") or {}

    def num(source: dict, key: str) -> float:
        value = source.get(key)
        return float(value) if isinstance(value, (int, float)) else 0.0

    topk_raw = num(ai, "topk_pattern_raw") or num(ai, "topk_pattern")
    topk_calibrated = num(ai, "topk_calibrated_risk")
    if topk_raw and not topk_calibrated:
        topk_calibrated = float(
            calibrate_topk_risk(
                topk_raw,
                eligible_sentence_count=3,
            ).get("topk_calibrated_risk", topk_raw)
        )

    return {
        "unsupported_claim_risk": num(writing, "unsupported_claim_risk"),
        "broad_claim_risk": num(writing, "broad_claim_risk"),
        "source_grounding_risk": num(writing, "source_grounding_risk"),
        "lived_detail_risk": num(writing, "lived_detail_risk"),
        "generic_assertion_risk": num(ai, "generic_assertion_risk"),
        "topk_pattern": topk_calibrated,
        "topk_pattern_raw": topk_raw,
        "topk_calibrated_risk": topk_calibrated,
        "predictability": num(ai, "predictability"),
    }


def _ai_footprint_profile(report_dict: dict | None) -> dict:
    """Build rewrite-only AI-footprint buckets from existing scanner fields."""
    if not isinstance(report_dict, dict):
        return {
            "authorship_footprint": {},
            "structural_footprint": {},
            "semantic_footprint": {},
            "grounding_footprint": {},
            "external_ai_flag_risk": 0.0,
        }
    badge = report_dict.get("ai_risk_badge") or {}
    ai_components = badge.get("ai_components") or {}
    writing_components = badge.get("writing_components") or {}
    integrity = _integrity_scores(report_dict)
    contribution = _contribution_scores(report_dict)

    def num(value, default=0.0) -> float:
        return float(value) if isinstance(value, (int, float)) else float(default)

    def feature(key: str, default=0.0) -> float:
        value = _feature_percent(report_dict, key)
        return num(value, default)

    topk_raw = num(ai_components.get("topk_pattern_raw"), num(ai_components.get("topk_pattern")))
    topk_calibrated = num(ai_components.get("topk_calibrated_risk"), -1.0)
    if topk_calibrated < 0.0:
        topk_calibrated = float(
            calibrate_topk_risk(
                topk_raw,
                eligible_sentence_count=3,
            ).get("topk_calibrated_risk", topk_raw)
        )

    authorship = {
        "ai_authorship": num(integrity.get("ai_authorship")),
        "ai_likelihood": feature("ai_likelihood", num(badge.get("ai_likelihood_score"))),
        "topk_pattern_raw": topk_raw,
        "topk_calibrated_risk": topk_calibrated,
        "topk_pattern": topk_raw,
        "predictability": num(ai_components.get("predictability")),
        "rewrite_smoothness": feature("rewrite_smoothness"),
    }
    structural = {
        "ai_transformation": num(contribution.get("ai_transformation")),
        "discourse_regularity": feature("discourse_regularity_risk"),
        "sentence_rhythm_uniformity": num(ai_components.get("sentence_rhythm_uniformity")),
        "paragraph_symmetry": num(ai_components.get("paragraph_symmetry")),
    }
    semantic = {
        "semantic_uniformity": feature("semantic_uniformity_risk"),
        "expansion_pattern": feature("outline_to_text_expansion"),
        "source_similarity": feature("source_similarity"),
        "surface_similarity": feature("surface_similarity"),
        "generic_assertion_risk": num(ai_components.get("generic_assertion_risk")),
        "qualifying_text_ai_density": num(ai_components.get("qualifying_text_ai_density")),
    }
    grounding = {
        "unsupported_claim_risk": num(writing_components.get("unsupported_claim_risk")),
        "broad_claim_risk": num(writing_components.get("broad_claim_risk")),
        "citation_weakness_risk": num(writing_components.get("citation_weakness_risk")),
        "source_grounding_risk": num(writing_components.get("source_grounding_risk")),
    }
    risk = (
        authorship["ai_authorship"] * 0.20
        + structural["ai_transformation"] * 0.18
        + authorship["ai_likelihood"] * 0.16
        + authorship["topk_calibrated_risk"] * 0.14
        + authorship["rewrite_smoothness"] * 0.12
        + semantic["semantic_uniformity"] * 0.08
        + semantic["qualifying_text_ai_density"] * 0.07
        + structural["discourse_regularity"] * 0.04
        + semantic["generic_assertion_risk"] * 0.03
        + grounding["unsupported_claim_risk"] * 0.015
        + grounding["broad_claim_risk"] * 0.005
    )
    return {
        "authorship_footprint": {key: round(value, 3) for key, value in authorship.items()},
        "structural_footprint": {key: round(value, 3) for key, value in structural.items()},
        "semantic_footprint": {key: round(value, 3) for key, value in semantic.items()},
        "grounding_footprint": {key: round(value, 3) for key, value in grounding.items()},
        "external_ai_flag_risk": round(risk, 3),
    }


def _ai_footprint_flatten(profile: dict | None) -> dict:
    merged = {}
    profile = profile or {}
    for bucket in (
        "authorship_footprint",
        "structural_footprint",
        "semantic_footprint",
        "grounding_footprint",
    ):
        values = profile.get(bucket) or {}
        if isinstance(values, dict):
            merged.update(values)
    merged["external_ai_flag_risk"] = profile.get("external_ai_flag_risk", 0.0)
    return merged


def _turnitin_like_ai_profile(report_dict: dict | None) -> dict:
    """Rewrite-only Turnitin-like score proxy from existing scanner fields."""
    return turnitin_like_ai_profile_from_report(report_dict)


def _turnitin_like_component_drops(before: dict | None, after: dict | None) -> dict:
    before_components = (before or {}).get("components") or {}
    after_components = (after or {}).get("components") or {}
    drops = {
        key: round(float(before_components.get(key, 0.0)) - float(after_components.get(key, 0.0)), 3)
        for key in TURNITIN_LIKE_COMPONENT_WEIGHTS
    }
    drops["human_anchor_suppression"] = round(
        float((after or {}).get("human_anchor_suppression", 0.0))
        - float((before or {}).get("human_anchor_suppression", 0.0)),
        3,
    )
    drops["turnitin_like_ai_score"] = round(
        float((before or {}).get("score", 0.0)) - float((after or {}).get("score", 0.0)),
        3,
    )
    return drops


def _remaining_turnitin_like_drivers(profile: dict | None) -> list[dict]:
    profile = profile if isinstance(profile, dict) else {}
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    weighted = profile.get("weighted_components") if isinstance(profile.get("weighted_components"), dict) else {}
    if bool(profile.get("target_met")):
        return []
    target_gap = float(profile.get("target_gap") or 0.0)
    remaining = [
        {
            "driver": key,
            "value": round(float(value), 3),
            "formula_weight": round(float(TURNITIN_LIKE_COMPONENT_WEIGHTS[key]), 3),
            "weighted_contribution": round(float(weighted.get(key, 0.0)), 3),
            "target_gap": round(target_gap, 3),
        }
        for key, value in components.items()
        if key in TURNITIN_LIKE_COMPONENT_WEIGHTS
        and isinstance(value, (int, float))
        and float(weighted.get(key, 0.0)) > 0.0
    ]
    suppression = profile.get("human_anchor_suppression")
    if isinstance(suppression, (int, float)) and float(suppression) < 45.0:
        remaining.append({
            "driver": "human_anchor_suppression",
            "value": round(float(suppression), 3),
            "target_direction": "increase",
            "available_suppression_headroom": round(max(0.0, 45.0 - float(suppression)), 3),
            "weighted_contribution": round(-float(suppression), 3),
            "target_gap": round(target_gap, 3),
        })
    remaining.sort(
        key=lambda row: (
            0 if row.get("driver") == "human_anchor_suppression" else 1,
            abs(float(row.get("weighted_contribution", 0.0))),
        ),
        reverse=True,
    )
    return remaining


def _turnitin_like_ai_gate_status(
    original_report: dict | None,
    candidate_report: dict | None,
    *,
    review_burden_delta: int | float = 0,
    weighted_severity_delta: int | float = 0,
    critical_high_delta: int | float = 0,
    ai_score_regressed: bool = False,
) -> dict:
    before = _turnitin_like_ai_profile(original_report)
    after = _turnitin_like_ai_profile(candidate_report)
    drops = _turnitin_like_component_drops(before, after)
    ai_before = _ai_footprint_flatten(_ai_footprint_profile(original_report))
    ai_after = _ai_footprint_flatten(_ai_footprint_profile(candidate_report))
    score_drop = float(drops.get("turnitin_like_ai_score") or 0.0)
    improvement_epsilon = 0.001
    target_score = TURNITIN_LIKE_TARGET_AI_SCORE
    major_backfire_limit = _float_env("DRAFTPROOF_TURNITIN_LIKE_MAJOR_COMPONENT_BACKFIRE", 8.0)
    component_backfires = [
        {
            "driver": key,
            "increase": round(abs(float(drop)), 3),
            "before": (before.get("components") or {}).get(key),
            "after": (after.get("components") or {}).get(key),
        }
        for key, drop in drops.items()
        if key in TURNITIN_LIKE_COMPONENT_WEIGHTS
        and isinstance(drop, (int, float))
        and float(drop) <= -major_backfire_limit
    ]
    if float(drops.get("human_anchor_suppression") or 0.0) <= -major_backfire_limit:
        component_backfires.append({
            "driver": "human_anchor_suppression",
            "increase": round(abs(float(drops.get("human_anchor_suppression") or 0.0)), 3),
            "before": before.get("human_anchor_suppression"),
            "after": after.get("human_anchor_suppression"),
        })
    ai_authorship_drop = float(ai_before.get("ai_authorship", 0.0)) - float(ai_after.get("ai_authorship", 0.0))
    ai_transformation_drop = float(ai_before.get("ai_transformation", 0.0)) - float(ai_after.get("ai_transformation", 0.0))
    safety_clean = bool(
        not ai_score_regressed
        and float(review_burden_delta or 0.0) <= 0.0
        and float(weighted_severity_delta or 0.0) <= 0.0
        and float(critical_high_delta or 0.0) <= 0.0
        and ai_authorship_drop >= 0.0
        and ai_transformation_drop >= 0.0
        and not component_backfires
    )
    improved = bool(score_drop > improvement_epsilon)
    score_target_met = bool(float(after.get("score", 100.0)) < target_score)
    achieved = bool(safety_clean and improved and score_target_met)
    partial = bool(safety_clean and improved and not achieved)
    if achieved:
        outcome = "ai_mitigated"
    elif partial:
        outcome = "partially_ai_mitigated"
    elif safety_clean and not improved and (
        float(review_burden_delta or 0.0) < 0.0
        or float(weighted_severity_delta or 0.0) < 0.0
    ):
        outcome = "cleanup_improved"
    else:
        outcome = "no_turnitin_like_improvement"
    return {
        "version": "turnitin_like_gate_v1",
        "before": before,
        "after": after,
        "score_before": before.get("score"),
        "score_after": after.get("score"),
        "score_drop": round(score_drop, 3),
        "component_drops": drops,
        "remaining_turnitin_like_drivers": _remaining_turnitin_like_drivers(after),
        "component_backfires": component_backfires,
        "ai_authorship_drop": round(ai_authorship_drop, 3),
        "ai_transformation_drop": round(ai_transformation_drop, 3),
        "safety_clean": safety_clean,
        "improved": improved,
        "improvement_epsilon": improvement_epsilon,
        "safe_band": achieved,
        "target_met": score_target_met,
        "target_score": round(float(target_score), 3),
        "target_gap": round(max(0.0, float(after.get("score", 100.0)) - float(target_score)), 3),
        "outcome_class": outcome,
        "thresholds": {
            "safe_band": round(float(target_score), 3),
            "target_score": round(float(target_score), 3),
            "improvement_epsilon": round(float(improvement_epsilon), 3),
            "major_component_backfire": round(float(major_backfire_limit), 3),
        },
    }


def _turnitin_like_candidate_rank(
    gate: dict | None,
    *,
    review_burden_delta: int | float = 0,
    weighted_severity_delta: int | float = 0,
    critical_high_delta: int | float = 0,
) -> tuple:
    gate = gate if isinstance(gate, dict) else {}
    drops = gate.get("component_drops") if isinstance(gate.get("component_drops"), dict) else {}
    return (
        1 if gate.get("safe_band") else 0,
        1 if gate.get("safety_clean") else 0,
        float(gate.get("score_drop") or 0.0),
        float(drops.get("ai_likelihood") or 0.0),
        float(drops.get("topk_calibrated_risk") or 0.0),
        float(drops.get("semantic_uniformity") or 0.0),
        float(drops.get("rewrite_smoothness") or 0.0),
        float(drops.get("patchwork_expansion") or 0.0),
        float(drops.get("signal_agreement") or 0.0),
        float(drops.get("human_anchor_suppression") or 0.0),
        -len(gate.get("component_backfires") or []),
        -max(0.0, float(review_burden_delta or 0.0)),
        -max(0.0, float(weighted_severity_delta or 0.0)),
        -max(0.0, float(critical_high_delta or 0.0)),
        -float((gate.get("after") or {}).get("score") or 100.0),
    )


def _formula_gap_changed_word_count(source_text: str, candidate_text: str) -> int:
    """Approximate changed-word budget for formula-drop efficiency reporting."""
    source_tokens = re.findall(r"\w+|[^\w\s]", source_text or "", flags=re.UNICODE)
    candidate_tokens = re.findall(r"\w+|[^\w\s]", candidate_text or "", flags=re.UNICODE)
    if not source_tokens and not candidate_tokens:
        return 0
    matcher = SequenceMatcher(None, source_tokens, candidate_tokens, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return int(changed)


def _formula_gap_weighted_driver_plan(profile: dict | None, *, safety_margin: float = 3.0) -> list[dict]:
    profile = profile if isinstance(profile, dict) else {}
    score = float(profile.get("score") or 0.0)
    target_score = float(profile.get("target_score") or TURNITIN_LIKE_TARGET_AI_SCORE)
    required = max(0.0, score - target_score + float(safety_margin or 0.0))
    components = profile.get("components") if isinstance(profile.get("components"), dict) else {}
    weighted = profile.get("weighted_components") if isinstance(profile.get("weighted_components"), dict) else {}
    plan: list[dict] = []
    remaining = required
    for driver, contribution in sorted(
        weighted.items(),
        key=lambda item: float(item[1]) if isinstance(item[1], (int, float)) else 0.0,
        reverse=True,
    ):
        if driver not in TURNITIN_LIKE_COMPONENT_WEIGHTS:
            continue
        weighted_value = max(0.0, float(contribution or 0.0))
        if weighted_value <= 0.0:
            continue
        required_weighted_drop = min(weighted_value, remaining) if remaining > 0.0 else 0.0
        weight = float(TURNITIN_LIKE_COMPONENT_WEIGHTS[driver])
        plan.append({
            "driver": driver,
            "component_value": round(float(components.get(driver) or 0.0), 3),
            "formula_weight": round(weight, 3),
            "weighted_contribution": round(weighted_value, 3),
            "required_weighted_drop": round(required_weighted_drop, 3),
            "required_raw_drop": round(required_weighted_drop / weight, 3) if weight > 0.0 else 0.0,
        })
        remaining = max(0.0, remaining - required_weighted_drop)
    suppression = float(profile.get("human_anchor_suppression") or 0.0)
    suppression_headroom = max(0.0, 45.0 - suppression)
    required_suppression_gain = min(suppression_headroom, remaining) if remaining > 0.0 else 0.0
    plan.append({
        "driver": "human_anchor_suppression",
        "component_value": round(suppression, 3),
        "formula_weight": -1.0,
        "weighted_contribution": round(-suppression, 3),
        "target_direction": "increase",
        "available_suppression_headroom": round(suppression_headroom, 3),
        "required_suppression_gain": round(required_suppression_gain, 3),
    })
    return plan


_FORMULA_DRIVER_CONTROL_PROFILES = {
    "ai_likelihood": {
        "strategy_family": "LIKELIHOOD_TEXTURE_REBUILD",
        "actionability": 0.78,
        "backfire_risk": 0.28,
        "typical_backfire": ["rewrite_smoothness", "semantic_uniformity"],
    },
    "topk_calibrated_risk": {
        "strategy_family": "TOPK_ROUTE_REBUILD",
        "actionability": 0.72,
        "backfire_risk": 0.34,
        "typical_backfire": ["semantic_drift", "rewrite_smoothness"],
    },
    "semantic_uniformity": {
        "strategy_family": "SEMANTIC_VARIANCE_RESTRUCTURE",
        "actionability": 0.64,
        "backfire_risk": 0.36,
        "typical_backfire": ["semantic_drift", "review_burden"],
    },
    "rewrite_smoothness": {
        "strategy_family": "SMOOTHNESS_DEPOLISH",
        "actionability": 0.70,
        "backfire_risk": 0.26,
        "typical_backfire": ["review_burden", "readability"],
    },
    "patchwork_expansion": {
        "strategy_family": "PATCHWORK_COLLAPSE",
        "actionability": 0.82,
        "backfire_risk": 0.22,
        "typical_backfire": ["anchor_loss", "semantic_drift"],
    },
    "signal_agreement": {
        "strategy_family": "SIGNAL_DISAGREEMENT_REBALANCE",
        "actionability": 0.48,
        "backfire_risk": 0.42,
        "typical_backfire": ["component_backfire"],
    },
    "human_anchor_suppression": {
        "strategy_family": "HUMAN_ANCHOR_SUPPRESSION_GAIN",
        "actionability": 0.62,
        "backfire_risk": 0.40,
        "typical_backfire": ["ai_authorship", "unsupported_claim_risk"],
    },
}


def _formula_gap_driver_priority_plan(
    before: dict | None,
    after: dict | None,
    *,
    safety_margin: float = 3.0,
) -> list[dict]:
    """Rank the next drivers by weighted impact, movable headroom, and risk cost."""
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    target_score = float(after.get("target_score") or before.get("target_score") or TURNITIN_LIKE_TARGET_AI_SCORE)
    remaining_gap = max(0.0, float(after.get("score") or 0.0) - target_score)
    gap_to_cover = remaining_gap + float(safety_margin or 0.0)
    before_weighted = before.get("weighted_components") if isinstance(before.get("weighted_components"), dict) else {}
    after_weighted = after.get("weighted_components") if isinstance(after.get("weighted_components"), dict) else {}
    after_components = after.get("components") if isinstance(after.get("components"), dict) else {}
    rows: list[dict] = []
    for driver, weight in TURNITIN_LIKE_COMPONENT_WEIGHTS.items():
        weighted_remaining = max(0.0, float(after_weighted.get(driver) or 0.0))
        if weighted_remaining <= 0.0:
            continue
        achieved_drop = float(before_weighted.get(driver) or 0.0) - weighted_remaining
        feasible_headroom = min(weighted_remaining, gap_to_cover) if gap_to_cover > 0.0 else 0.0
        profile = _FORMULA_DRIVER_CONTROL_PROFILES.get(driver, {})
        actionability = float(profile.get("actionability", 0.5))
        backfire_risk = float(profile.get("backfire_risk", 0.35))
        priority = feasible_headroom * (0.65 + actionability) * max(0.15, 1.0 - backfire_risk * 0.45)
        rows.append({
            "driver": driver,
            "strategy_family": profile.get("strategy_family", driver.upper()),
            "raw_value_after": round(float(after_components.get(driver) or 0.0), 3),
            "formula_weight": round(float(weight), 3),
            "weighted_remaining": round(weighted_remaining, 3),
            "achieved_weighted_drop": round(achieved_drop, 3),
            "feasible_weighted_headroom": round(feasible_headroom, 3),
            "remaining_gap_share": round(
                feasible_headroom / max(remaining_gap, 1.0),
                3,
            ),
            "actionability": round(actionability, 3),
            "backfire_risk": round(backfire_risk, 3),
            "priority_score": round(priority, 3),
            "expected_net_gain": round(priority, 3),
            "typical_backfire": profile.get("typical_backfire", []),
            "control_goal": "reduce",
        })

    suppression = float(after.get("human_anchor_suppression") or 0.0)
    suppression_before = float(before.get("human_anchor_suppression") or 0.0)
    suppression_headroom = max(0.0, 45.0 - suppression)
    if suppression_headroom > 0.0:
        profile = _FORMULA_DRIVER_CONTROL_PROFILES["human_anchor_suppression"]
        feasible_headroom = min(suppression_headroom, gap_to_cover) if gap_to_cover > 0.0 else 0.0
        actionability = float(profile.get("actionability", 0.5))
        backfire_risk = float(profile.get("backfire_risk", 0.35))
        priority = feasible_headroom * (0.65 + actionability) * max(0.15, 1.0 - backfire_risk * 0.45)
        rows.append({
            "driver": "human_anchor_suppression",
            "strategy_family": profile.get("strategy_family"),
            "raw_value_after": round(suppression, 3),
            "formula_weight": -1.0,
            "weighted_remaining": round(-suppression, 3),
            "achieved_weighted_drop": round(suppression - suppression_before, 3),
            "feasible_weighted_headroom": round(feasible_headroom, 3),
            "remaining_gap_share": round(
                feasible_headroom / max(remaining_gap, 1.0),
                3,
            ),
            "actionability": round(actionability, 3),
            "backfire_risk": round(backfire_risk, 3),
            "priority_score": round(priority, 3),
            "expected_net_gain": round(priority, 3),
            "typical_backfire": profile.get("typical_backfire", []),
            "control_goal": "increase",
        })
    rows.sort(
        key=lambda row: (
            float(row.get("priority_score", 0.0)),
            float(row.get("feasible_weighted_headroom", 0.0)),
            float(row.get("weighted_remaining", 0.0)),
        ),
        reverse=True,
    )
    return rows


def _formula_observed_driver_movement(candidates: list[dict] | tuple[dict, ...] | None) -> dict:
    """Summarize measured formula-driver movement across the current candidate frontier."""
    drivers = list(TURNITIN_LIKE_COMPONENT_WEIGHTS) + ["human_anchor_suppression"]
    observed = {
        driver: {
            "candidate_count": 0,
            "safe_candidate_count": 0,
            "best_observed_drop": 0.0,
            "best_observed_strategy": None,
            "best_safe_drop": 0.0,
            "best_safe_strategy": None,
            "best_blocked_drop": 0.0,
            "best_blocked_strategy": None,
            "best_blocked_reason": None,
        }
        for driver in drivers
    }
    for row in candidates or []:
        if not isinstance(row, dict):
            continue
        contract = row.get("formula_gap_contract")
        if not isinstance(contract, dict):
            contract = ((row.get("selection_status") or {}).get("formula_gap_contract") or {})
        drops = contract.get("weighted_driver_drops") if isinstance(contract.get("weighted_driver_drops"), dict) else {}
        status = row.get("selection_status") if isinstance(row.get("selection_status"), dict) else {}
        gate = row.get("turnitin_like_ai_gate") if isinstance(row.get("turnitin_like_ai_gate"), dict) else {}
        if not gate:
            gate = status.get("turnitin_like_ai_gate") if isinstance(status.get("turnitin_like_ai_gate"), dict) else {}
        selectable = bool(status.get("selectable"))
        safety_clean = bool(gate.get("safety_clean", selectable))
        safe = selectable and safety_clean
        reason = row.get("reason") or status.get("reason")
        strategy = row.get("strategy") or status.get("strategy")
        for driver in drivers:
            driver_drop = drops.get(driver) if isinstance(drops.get(driver), dict) else {}
            if not isinstance(driver_drop, dict):
                continue
            value = driver_drop.get("drop", driver_drop.get("gain"))
            if not isinstance(value, (int, float)):
                continue
            value = round(float(value), 3)
            if value <= 0.0:
                continue
            item = observed[driver]
            item["candidate_count"] += 1
            if value > float(item["best_observed_drop"] or 0.0):
                item["best_observed_drop"] = value
                item["best_observed_strategy"] = strategy
            if safe:
                item["safe_candidate_count"] += 1
                if value > float(item["best_safe_drop"] or 0.0):
                    item["best_safe_drop"] = value
                    item["best_safe_strategy"] = strategy
            elif value > float(item["best_blocked_drop"] or 0.0):
                item["best_blocked_drop"] = value
                item["best_blocked_strategy"] = strategy
                item["best_blocked_reason"] = reason
    return observed


def _formula_portfolio_plan_from_profiles(
    before: dict | None,
    after: dict | None,
    *,
    observed_driver_movement: dict | None = None,
    safety_margin: float = 3.0,
) -> dict:
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    target_score = float(after.get("target_score") or before.get("target_score") or TURNITIN_LIKE_TARGET_AI_SCORE)
    score_before = float(before.get("score") or 0.0)
    score_after = float(after.get("score") or 0.0)
    positive_before = float(before.get("raw_positive_score") or 0.0)
    positive_after = float(after.get("raw_positive_score") or 0.0)
    suppression_before = float(before.get("human_anchor_suppression") or 0.0)
    suppression_after = float(after.get("human_anchor_suppression") or 0.0)
    suppression_headroom = max(0.0, 45.0 - suppression_after)
    remaining_gap = max(0.0, score_after - target_score)
    required_total_gain = remaining_gap + float(safety_margin or 0.0)
    required_suppression_gain = min(suppression_headroom, required_total_gain)
    observed_driver_movement = observed_driver_movement if isinstance(observed_driver_movement, dict) else {}
    base_priorities = _formula_gap_driver_priority_plan(before, after, safety_margin=safety_margin)
    driver_priorities: list[dict] = []
    for row in base_priorities:
        if not isinstance(row, dict):
            continue
        driver = str(row.get("driver") or "")
        movement = observed_driver_movement.get(driver) if isinstance(observed_driver_movement.get(driver), dict) else {}
        headroom = float(row.get("feasible_weighted_headroom") or 0.0)
        static_actionability = float(row.get("actionability") or 0.0)
        safe_drop = float(movement.get("best_safe_drop") or 0.0)
        blocked_drop = float(movement.get("best_blocked_drop") or 0.0)
        observed_safe_ratio = min(1.0, safe_drop / max(headroom, 1.0))
        observed_blocked_ratio = min(1.0, blocked_drop / max(headroom, 1.0))
        observed_actionability = (
            min(1.0, static_actionability * 0.65 + observed_safe_ratio * 0.35)
            if movement.get("candidate_count")
            else static_actionability
        )
        if safe_drop <= 0.0 and blocked_drop > 0.0:
            observed_actionability = min(1.0, static_actionability * 0.55 + observed_blocked_ratio * 0.20)
        backfire_risk = float(row.get("backfire_risk") or 0.0)
        if blocked_drop > safe_drop and blocked_drop > 0.0:
            backfire_risk = min(1.0, backfire_risk + 0.12)
        expected_net_gain = headroom * (0.65 + observed_actionability) * max(0.10, 1.0 - backfire_risk * 0.50)
        enriched = dict(row)
        enriched.update({
            "static_actionability": round(static_actionability, 3),
            "observed_actionability": round(observed_actionability, 3),
            "observed_safe_drop": round(safe_drop, 3),
            "observed_best_drop": round(float(movement.get("best_observed_drop") or 0.0), 3),
            "observed_blocked_drop": round(blocked_drop, 3),
            "observed_candidate_count": int(movement.get("candidate_count") or 0),
            "observed_safe_candidate_count": int(movement.get("safe_candidate_count") or 0),
            "best_safe_strategy": movement.get("best_safe_strategy"),
            "best_blocked_strategy": movement.get("best_blocked_strategy"),
            "best_blocked_reason": movement.get("best_blocked_reason"),
            "backfire_risk": round(backfire_risk, 3),
            "expected_net_gain": round(expected_net_gain, 3),
            "priority_score": round(expected_net_gain, 3),
        })
        driver_priorities.append(enriched)
    driver_priorities.sort(
        key=lambda item: (
            float(item.get("expected_net_gain", 0.0)),
            float(item.get("feasible_weighted_headroom", 0.0)),
            float(item.get("observed_safe_drop", 0.0)),
        ),
        reverse=True,
    )
    selected_portfolio = []
    cumulative = 0.0
    for row in driver_priorities:
        if cumulative >= required_total_gain:
            break
        selected_portfolio.append({
            "driver": row.get("driver"),
            "strategy_family": row.get("strategy_family"),
            "expected_net_gain": row.get("expected_net_gain"),
            "control_goal": row.get("control_goal"),
        })
        cumulative += float(row.get("expected_net_gain") or 0.0)
    if remaining_gap > 0.0 and suppression_headroom > 0.0 and not any(
        item.get("driver") == "human_anchor_suppression" for item in selected_portfolio
    ):
        anchor_row = next(
            (
                row for row in driver_priorities
                if row.get("driver") == "human_anchor_suppression"
            ),
            None,
        )
        if anchor_row:
            selected_portfolio.insert(1 if selected_portfolio else 0, {
                "driver": anchor_row.get("driver"),
                "strategy_family": anchor_row.get("strategy_family"),
                "expected_net_gain": anchor_row.get("expected_net_gain"),
                "control_goal": anchor_row.get("control_goal"),
                "required_because": "subtractive Human Anchor suppression has usable headroom",
            })
    return {
        "version": "formula_portfolio_plan_v1",
        "score_before": round(score_before, 3),
        "score_after": round(score_after, 3),
        "target_score": round(target_score, 3),
        "target_met": bool(score_after < target_score),
        "required_gap": round(remaining_gap, 3),
        "safety_margin": round(float(safety_margin or 0.0), 3),
        "required_total_gain": round(required_total_gain, 3),
        "positive_ai_burden": {
            "before": round(positive_before, 3),
            "after": round(positive_after, 3),
            "drop": round(positive_before - positive_after, 3),
        },
        "human_anchor_suppression": {
            "before": round(suppression_before, 3),
            "after": round(suppression_after, 3),
            "gain": round(suppression_after - suppression_before, 3),
        },
        "suppression_headroom": round(suppression_headroom, 3),
        "required_suppression_gain": round(required_suppression_gain, 3),
        "driver_priorities": driver_priorities,
        "selected_driver_portfolio": selected_portfolio,
        "expected_net_gain": {
            str(row.get("driver")): row.get("expected_net_gain")
            for row in driver_priorities
            if row.get("driver")
        },
        "observed_driver_movement": observed_driver_movement,
    }


def _formula_portfolio_plan(
    original_report: dict | None,
    candidate_report: dict | None,
    *,
    observed_candidates: list[dict] | tuple[dict, ...] | None = None,
    safety_margin: float = 3.0,
) -> dict:
    before = _turnitin_like_ai_profile(original_report)
    after = _turnitin_like_ai_profile(candidate_report)
    observed = _formula_observed_driver_movement(observed_candidates)
    return _formula_portfolio_plan_from_profiles(
        before,
        after,
        observed_driver_movement=observed,
        safety_margin=safety_margin,
    )


def _formula_gap_contract(
    original_report: dict | None,
    candidate_report: dict | None,
    *,
    source_text: str = "",
    candidate_text: str = "",
    safety_margin: float = 3.0,
) -> dict:
    """Rewrite controller contract for closing the shared Turnitin-like formula gap."""
    before = _turnitin_like_ai_profile(original_report)
    after = _turnitin_like_ai_profile(candidate_report)
    component_drops = _turnitin_like_component_drops(before, after)
    before_weighted = before.get("weighted_components") if isinstance(before.get("weighted_components"), dict) else {}
    after_weighted = after.get("weighted_components") if isinstance(after.get("weighted_components"), dict) else {}
    weighted_driver_drops: dict[str, dict] = {}
    weighted_formula_drop = 0.0
    for driver in TURNITIN_LIKE_COMPONENT_WEIGHTS:
        before_value = float(before_weighted.get(driver) or 0.0)
        after_value = float(after_weighted.get(driver) or 0.0)
        drop = before_value - after_value
        weighted_formula_drop += drop
        weighted_driver_drops[driver] = {
            "before": round(before_value, 3),
            "after": round(after_value, 3),
            "drop": round(drop, 3),
            "raw_before": (before.get("components") or {}).get(driver),
            "raw_after": (after.get("components") or {}).get(driver),
            "raw_drop": round(float(component_drops.get(driver) or 0.0), 3),
            "formula_weight": round(float(TURNITIN_LIKE_COMPONENT_WEIGHTS[driver]), 3),
        }
    suppression_gain = float(component_drops.get("human_anchor_suppression") or 0.0)
    weighted_formula_drop += suppression_gain
    weighted_driver_drops["human_anchor_suppression"] = {
        "before": round(float(before.get("human_anchor_suppression") or 0.0), 3),
        "after": round(float(after.get("human_anchor_suppression") or 0.0), 3),
        "gain": round(suppression_gain, 3),
        "drop": round(suppression_gain, 3),
        "formula_weight": -1.0,
        "target_direction": "increase",
    }
    score_before = float(before.get("score") or 0.0)
    score_after = float(after.get("score") or 0.0)
    measured_score_drop = score_before - score_after
    changed_words = _formula_gap_changed_word_count(source_text, candidate_text) if source_text or candidate_text else 0
    efficiency = measured_score_drop / max(1, changed_words)
    target_score = float(after.get("target_score") or before.get("target_score") or TURNITIN_LIKE_TARGET_AI_SCORE)
    remaining_gap = max(0.0, score_after - target_score)
    portfolio = _formula_portfolio_plan_from_profiles(before, after, safety_margin=safety_margin)
    contract = {
        "version": "formula_gap_contract_v1",
        "score_before": round(score_before, 3),
        "score_after": round(score_after, 3),
        "score_drop": round(measured_score_drop, 3),
        "weighted_formula_score_drop": round(weighted_formula_drop, 3),
        "target_score": round(target_score, 3),
        "target_gap_before": round(max(0.0, score_before - target_score), 3),
        "target_gap": round(remaining_gap, 3),
        "remaining_formula_gap": round(remaining_gap, 3),
        "target_met": bool(score_after < target_score),
        "weighted_contributions_before": {
            key: round(float(value or 0.0), 3)
            for key, value in before_weighted.items()
            if key in TURNITIN_LIKE_COMPONENT_WEIGHTS
        },
        "weighted_contributions_after": {
            key: round(float(value or 0.0), 3)
            for key, value in after_weighted.items()
            if key in TURNITIN_LIKE_COMPONENT_WEIGHTS
        },
        "positive_ai_burden": portfolio.get("positive_ai_burden"),
        "human_anchor_suppression": portfolio.get("human_anchor_suppression"),
        "suppression_headroom": portfolio.get("suppression_headroom"),
        "required_suppression_gain": portfolio.get("required_suppression_gain"),
        "formula_portfolio_plan": portfolio,
        "weighted_driver_plan": _formula_gap_weighted_driver_plan(before, safety_margin=safety_margin),
        "driver_priority_plan": portfolio.get("driver_priorities") or [],
        "weighted_driver_drops": weighted_driver_drops,
        "component_drops": component_drops,
        "changed_word_count": changed_words,
        "weighted_driver_drop_efficiency": round(efficiency, 6),
        "remaining_formula_drivers": _remaining_turnitin_like_drivers(after),
    }
    contract["next_formula_driver"] = (
        (contract.get("driver_priority_plan") or [{}])[0].get("driver")
        if contract.get("driver_priority_plan") else None
    )
    contract["priority_basis"] = (
        "weighted contribution plus feasible headroom plus actionability minus backfire risk"
    )
    contract["why_not_below_20"] = (
        "turnitin-like formula target achieved"
        if contract["target_met"]
        else (
            "Remaining weighted gap "
            f"{contract['remaining_formula_gap']} to target {contract['target_score']}; "
            "dominant drivers: "
            + ", ".join(
                str(row.get("driver"))
                for row in contract["remaining_formula_drivers"][:4]
                if isinstance(row, dict) and row.get("driver")
            )
        )
    )
    return contract


def _formula_gap_candidate_rank(contract: dict | None, gate: dict | None = None) -> tuple:
    contract = contract if isinstance(contract, dict) else {}
    gate = gate if isinstance(gate, dict) else {}
    drops = contract.get("weighted_driver_drops") if isinstance(contract.get("weighted_driver_drops"), dict) else {}

    def drop(driver: str) -> float:
        row = drops.get(driver) if isinstance(drops.get(driver), dict) else {}
        return float(row.get("drop") or row.get("gain") or 0.0)

    return (
        1 if gate.get("safety_clean", True) else 0,
        1 if contract.get("target_met") else 0,
        float(contract.get("score_drop") or 0.0),
        float(((contract.get("positive_ai_burden") or {}).get("drop")) or 0.0),
        drop("ai_likelihood"),
        drop("topk_calibrated_risk"),
        drop("semantic_uniformity"),
        drop("rewrite_smoothness"),
        drop("patchwork_expansion"),
        drop("signal_agreement"),
        drop("human_anchor_suppression"),
        float(contract.get("weighted_driver_drop_efficiency") or 0.0),
        -float(contract.get("score_after") if isinstance(contract.get("score_after"), (int, float)) else 100.0),
    )
