"""AI-search candidate classification and ranking policy.

The AI-search pipeline produces different kinds of "safe progress":
detector-driver movement, formula-only movement, and cleanup.  Treating all of
them as the same `selectable=True` state lets cleanup beat real mitigation.  This
module keeps those lanes separate before ranking candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .progress_policy import meaningful_ai_progress_gate


CLASS_REJECT = "reject"
CLASS_FALLBACK = "fallback"
CLASS_CLEANUP_ONLY = "cleanup_only"
CLASS_FORMULA_PROGRESS = "formula_progress"
CLASS_DETECTOR_PROGRESS = "detector_progress"
CLASS_DETECTOR_SAFE = "detector_safe"

CLASS_PRIORITY = {
    CLASS_REJECT: 0,
    CLASS_FALLBACK: 1,
    CLASS_CLEANUP_ONLY: 2,
    CLASS_FORMULA_PROGRESS: 3,
    CLASS_DETECTOR_PROGRESS: 4,
    CLASS_DETECTOR_SAFE: 5,
}

CORE_DRIVER_PROGRESS_FLOORS = {
    "ai_likelihood_drop": 2.0,
    "topk_calibrated_risk_drop": 2.0,
    "ai_authorship_drop": 2.0,
    "qualifying_text_ai_density_drop": 3.0,
    "rewrite_smoothness_drop": 2.0,
}


@dataclass(frozen=True)
class CandidateDecision:
    """Visible selector decision record for a rescanned candidate."""

    selectable: bool
    candidate_class: str
    reason: str
    rank: tuple
    formula_score_drop: float
    headline_ai_drop: float
    target_ai_score: float | None
    ai_target_gap: float | None
    detector_driver_drop_score: float
    target_met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "selectable": self.selectable,
            "class": self.candidate_class,
            "reason": self.reason,
            "rank": list(self.rank),
            "formula_score_drop": self.formula_score_drop,
            "headline_ai_drop": self.headline_ai_drop,
            "target_ai_score": self.target_ai_score,
            "ai_target_gap": self.ai_target_gap,
            "detector_driver_drop_score": self.detector_driver_drop_score,
            "target_met": self.target_met,
        }


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) else float(default)


def _optional_num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _authenticity_gate(status: dict) -> dict:
    gate = status.get("authenticity_gate")
    return gate if isinstance(gate, dict) else status


def _footprint_gate(status: dict) -> dict:
    gate = status.get("ai_footprint_gate")
    return gate if isinstance(gate, dict) else {}


def _formula_contract(status: dict, candidate_eval: dict | None = None) -> dict:
    eval_data = candidate_eval if isinstance(candidate_eval, dict) else {}
    contract = status.get("formula_gap_contract")
    if isinstance(contract, dict):
        return contract
    contract = eval_data.get("formula_gap_contract")
    return contract if isinstance(contract, dict) else {}


def _turnitin_gate(status: dict) -> dict:
    gate = status.get("turnitin_like_ai_gate")
    return gate if isinstance(gate, dict) else {}


def _eligible_span_density_gate(status: dict) -> dict:
    gate = status.get("eligible_span_density_gate")
    return gate if isinstance(gate, dict) else {}


def _human_shift_components(status: dict) -> dict:
    gate = _authenticity_gate(status)
    components = status.get("human_shift_components")
    if isinstance(components, dict):
        return components
    components = gate.get("human_shift_components")
    return components if isinstance(components, dict) else {}


def detector_progress_metrics(selection_status: dict | None) -> dict[str, float]:
    status = selection_status if isinstance(selection_status, dict) else {}
    gate = _authenticity_gate(status)
    footprint = _footprint_gate(status)
    drops = footprint.get("drops") if isinstance(footprint.get("drops"), dict) else {}
    components = _human_shift_components(status)
    ai_likelihood_drop = _num(drops.get("ai_likelihood"))
    topk_drop = _num(drops.get("topk_calibrated_risk"))
    authorship_drop = _num(gate.get("ai_authorship_delta"))
    external_drop = _num(drops.get("external_ai_flag_risk"))
    density_drop = _num(drops.get("qualifying_text_ai_density"))
    smoothness_drop = _num(components.get("rewrite_smoothness_reduction"))
    driver_drop_score = (
        max(0.0, ai_likelihood_drop) * 4.0
        + max(0.0, topk_drop) * 3.0
        + max(0.0, authorship_drop) * 2.0
        + max(0.0, density_drop) * 2.0
        + max(0.0, external_drop) * 1.5
        + max(0.0, smoothness_drop)
    )
    return {
        "ai_likelihood_drop": ai_likelihood_drop,
        "topk_calibrated_risk_drop": topk_drop,
        "ai_authorship_drop": authorship_drop,
        "external_ai_flag_risk_drop": external_drop,
        "qualifying_text_ai_density_drop": density_drop,
        "rewrite_smoothness_drop": smoothness_drop,
        "driver_drop_score": driver_drop_score,
    }


def classify_ai_search_candidate(selection_status: dict | None, candidate_eval: dict | None = None) -> dict[str, Any]:
    status = selection_status if isinstance(selection_status, dict) else {}
    if not status.get("selectable"):
        return {
            "class": CLASS_REJECT,
            "priority": CLASS_PRIORITY[CLASS_REJECT],
            "reason": status.get("reason") or "candidate_not_selectable",
            "detector_progress": detector_progress_metrics(status),
        }

    footprint = _footprint_gate(status)
    footprint_outcome = str(
        status.get("ai_footprint_outcome_class")
        or footprint.get("outcome_class")
        or ""
    )
    turnitin = _turnitin_gate(status)
    density_gate = _eligible_span_density_gate(status)
    formula = _formula_contract(status, candidate_eval)
    metrics = detector_progress_metrics(status)
    target_met = bool(formula.get("target_met") or turnitin.get("safe_band"))
    density_gate_present = bool(density_gate)
    density_safe = bool(not density_gate_present or density_gate.get("safe"))
    positive_burden = formula.get("positive_ai_burden") if isinstance(formula.get("positive_ai_burden"), dict) else {}
    meaningful_progress = meaningful_ai_progress_gate(
        turnitin_like_ai_score_drop=_num(formula.get("score_drop"), _num(turnitin.get("score_drop"))),
        ai_authorship_drop=metrics.get("ai_authorship_drop"),
        positive_ai_burden_drop=positive_burden.get("drop"),
        unsafe_eligible_density_drop=density_gate.get("unsafe_eligible_word_ratio_drop"),
        ai_window_vote_ratio_drop=density_gate.get("ai_sentence_vote_ratio_drop"),
    )
    core_driver_progress = any(
        max(0.0, float(metrics.get(name, 0.0))) >= floor
        for name, floor in CORE_DRIVER_PROGRESS_FLOORS.items()
    )
    real_detector_progress = bool(
        status.get("topk_safe_band_achieved")
        or status.get("ai_footprint_mitigation")
        or status.get("partial_ai_footprint_mitigation")
        or status.get("topk_blocker_progress")
        or footprint_outcome in {"ai_mitigated", "partially_ai_mitigated", "ai_footprint_blocked_by_texture"}
        or meaningful_progress.get("passed")
        or core_driver_progress
    )

    if target_met and density_safe and (status.get("turnitin_like_mitigation") or footprint_outcome == "ai_mitigated"):
        cls = CLASS_DETECTOR_SAFE
    elif real_detector_progress:
        cls = CLASS_DETECTOR_PROGRESS
    elif status.get("partial_turnitin_like_mitigation") or (
        turnitin.get("improved") and turnitin.get("safety_clean")
    ):
        cls = CLASS_FORMULA_PROGRESS
    elif (
        footprint_outcome == "cleanup_improved"
        or status.get("safe_partial_quality_improvement")
        or status.get("score_drag_removal")
    ):
        cls = CLASS_CLEANUP_ONLY
    else:
        cls = CLASS_FALLBACK

    return {
        "class": cls,
        "priority": CLASS_PRIORITY[cls],
        "reason": cls,
        "detector_progress": metrics,
        "meaningful_ai_progress_gate": meaningful_progress,
        "footprint_outcome": footprint_outcome,
        "target_met": target_met,
        "eligible_span_density_safe": density_safe,
    }


def detector_progress_rank(selection_status: dict | None, candidate_eval: dict | None = None) -> tuple:
    classification = classify_ai_search_candidate(selection_status, candidate_eval)
    metrics = classification["detector_progress"]
    return (
        int(classification["priority"]),
        float(metrics["driver_drop_score"]),
        max(0.0, float(metrics["ai_likelihood_drop"])),
        max(0.0, float(metrics["topk_calibrated_risk_drop"])),
        max(0.0, float(metrics["ai_authorship_drop"])),
        max(0.0, float(metrics["external_ai_flag_risk_drop"])),
    )


def ai_search_candidate_rank(
    selection_status: dict | None,
    candidate_eval: dict | None,
    *,
    candidate_ai: Any = None,
    candidate_review_burden: int | float = 0,
    candidate_weighted_severity: int | float = 0,
    candidate_finding_total: int | float = 0,
    original_review_burden: int | float = 0,
    original_weighted_severity: int | float = 0,
    original_finding_total: int | float = 0,
    target_human: float = 80.0,
    stage_target: float = 80.0,
    formula_gap_rank: tuple = (),
) -> tuple:
    status = selection_status if isinstance(selection_status, dict) else {}
    eval_data = candidate_eval if isinstance(candidate_eval, dict) else {}
    gate = _authenticity_gate(status)
    components = _human_shift_components(status)
    turnitin = _turnitin_gate(status)
    turnitin_drops = turnitin.get("component_drops") if isinstance(turnitin.get("component_drops"), dict) else {}
    formula = _formula_contract(status, eval_data)
    classification = classify_ai_search_candidate(status, eval_data)
    detector_rank = detector_progress_rank(status, eval_data)
    anchor_contract = status.get("human_anchor_driver_contract")
    if not isinstance(anchor_contract, dict):
        anchor_contract = eval_data.get("human_anchor_driver_contract")
    anchor_contract = anchor_contract if isinstance(anchor_contract, dict) else {}
    anchor_deltas = anchor_contract.get("deltas") if isinstance(anchor_contract.get("deltas"), dict) else {}
    multi_signal = status.get("multi_signal_contract")
    if not isinstance(multi_signal, dict):
        multi_signal = eval_data.get("multi_signal_contract")
    multi_signal = multi_signal if isinstance(multi_signal, dict) else {}

    human_shift = _num(status.get("human_shift_score", gate.get("human_shift_score")), -9999.0)
    human_delta = _num(gate.get("human_delta"), -9999.0)
    candidate_human = _num(gate.get("candidate_human", eval_data.get("human_contribution")), -9999.0)
    ai_authorship_delta = _num(gate.get("ai_authorship_delta"), -9999.0)
    ai_transform_delta = _num(gate.get("ai_transformation_delta"), -9999.0)
    reference_ai = _optional_num(status.get("reference_ai"))
    actual_ai = _optional_num(candidate_ai)
    target_ai = _optional_num(status.get("target_ai_score"))
    if target_ai is None:
        target_ai = _optional_num(status.get("target"))
    headline_ai_drop = (
        reference_ai - actual_ai
        if reference_ai is not None and actual_ai is not None
        else 0.0
    )
    ai_target_gap = (
        max(0.0, actual_ai - target_ai)
        if actual_ai is not None and target_ai is not None
        else 9999.0
    )
    headline_ai_not_worse = (
        reference_ai is None
        or actual_ai is None
        or headline_ai_drop >= 0.0
    )
    ai_drop_success = bool(
        status.get("ai_drop_success")
        or status.get("success")
        or formula.get("target_met")
        or turnitin.get("safe_band")
    )
    review_reduction = _num(original_review_burden) - _num(candidate_review_burden)
    severity_reduction = _num(original_weighted_severity) - _num(candidate_weighted_severity)
    finding_reduction = _num(original_finding_total) - _num(candidate_finding_total)
    severe_backfire_count = len(multi_signal.get("severe_backfires") or [])
    measured_score_drop = _num(formula.get("score_drop"), _num(turnitin.get("score_drop")))

    return (
        1 if status.get("selectable") else 0,
        1 if formula.get("target_met") else 0,
        1 if ai_drop_success else 0,
        1 if headline_ai_not_worse else 0,
        -ai_target_gap,
        headline_ai_drop,
        1 if human_shift >= 0 else 0,
        human_shift,
        -_num(candidate_ai, 9999.0),
        detector_rank,
        int(classification["priority"]),
        measured_score_drop,
        tuple(formula_gap_rank or ()),
        _num(formula.get("weighted_driver_drop_efficiency")),
        5 if turnitin.get("safe_band") else 3 if turnitin.get("improved") and turnitin.get("safety_clean") else 0,
        _num(turnitin.get("score_drop")),
        _num(turnitin_drops.get("ai_likelihood")),
        _num(turnitin_drops.get("topk_calibrated_risk")),
        _num(turnitin_drops.get("semantic_uniformity")),
        _num(turnitin_drops.get("rewrite_smoothness")),
        _num(turnitin_drops.get("patchwork_expansion")),
        _num(turnitin_drops.get("human_anchor_suppression")),
        _num(anchor_deltas.get("lived_detail_risk")),
        _num(anchor_deltas.get("human_anchor_score")),
        -severe_backfire_count,
        _num(multi_signal.get("balance_score")),
        ai_authorship_delta,
        _num(components.get("rewrite_smoothness_reduction")),
        1 if candidate_human >= target_human else 0,
        1 if candidate_human >= stage_target else 0,
        1 if human_shift > 0 else 0,
        human_delta,
        ai_transform_delta,
        _num(components.get("semantic_uniformity_reduction")),
        review_reduction,
        severity_reduction,
        finding_reduction,
    )


def build_candidate_decision(
    selection_status: dict | None,
    candidate_eval: dict | None,
    *,
    candidate_ai: Any = None,
    candidate_review_burden: int | float = 0,
    candidate_weighted_severity: int | float = 0,
    candidate_finding_total: int | float = 0,
    original_review_burden: int | float = 0,
    original_weighted_severity: int | float = 0,
    original_finding_total: int | float = 0,
    target_human: float = 80.0,
    stage_target: float = 80.0,
    formula_gap_rank: tuple = (),
) -> CandidateDecision:
    """Build the same selector rank plus explainable headline fields."""

    status = selection_status if isinstance(selection_status, dict) else {}
    eval_data = candidate_eval if isinstance(candidate_eval, dict) else {}
    classification = classify_ai_search_candidate(status, eval_data)
    metrics = detector_progress_metrics(status)
    formula = _formula_contract(status, eval_data)
    turnitin = _turnitin_gate(status)
    rank = ai_search_candidate_rank(
        status,
        eval_data,
        candidate_ai=candidate_ai,
        candidate_review_burden=candidate_review_burden,
        candidate_weighted_severity=candidate_weighted_severity,
        candidate_finding_total=candidate_finding_total,
        original_review_burden=original_review_burden,
        original_weighted_severity=original_weighted_severity,
        original_finding_total=original_finding_total,
        target_human=target_human,
        stage_target=stage_target,
        formula_gap_rank=formula_gap_rank,
    )
    reference_ai = _optional_num(status.get("reference_ai"))
    actual_ai = _optional_num(candidate_ai)
    target_ai = _optional_num(status.get("target_ai_score"))
    if target_ai is None:
        target_ai = _optional_num(status.get("target"))
    headline_ai_drop = (
        reference_ai - actual_ai
        if reference_ai is not None and actual_ai is not None
        else 0.0
    )
    ai_target_gap = (
        max(0.0, actual_ai - target_ai)
        if actual_ai is not None and target_ai is not None
        else None
    )
    return CandidateDecision(
        selectable=bool(status.get("selectable")),
        candidate_class=str(classification.get("class") or CLASS_REJECT),
        reason=str(status.get("reason") or classification.get("reason") or ""),
        rank=rank,
        formula_score_drop=_num(formula.get("score_drop"), _num(turnitin.get("score_drop"))),
        headline_ai_drop=float(headline_ai_drop),
        target_ai_score=target_ai,
        ai_target_gap=ai_target_gap,
        detector_driver_drop_score=float(metrics.get("driver_drop_score") or 0.0),
        target_met=bool(formula.get("target_met") or turnitin.get("safe_band")),
    )
