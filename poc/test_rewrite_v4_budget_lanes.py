from rewrite_v4.budget_lanes import (
    BudgetLane,
    budget_lanes_for_unit,
    budget_profile_passes,
    edit_budget_profile,
)
from rewrite_v4.experiment import _is_material_budget_lane_positive, _is_safe_budget_lane_positive


def test_v4_budget_profile_counts_changed_source_and_net_drift():
    source = (
        "Today’s education system is changing faster than many schools can comfortably manage. "
        "In the past, education was mostly built around the classroom, the textbook, and the teacher."
    )
    replacement = (
        "Today’s education system is changing faster than many schools can easily handle. "
        "Previously, education was mostly centred on the classroom, the textbook, and the teacher."
    )

    profile = edit_budget_profile(source, replacement)

    assert profile["source_words"] == len(source.split())
    assert profile["replacement_words"] == len(replacement.split())
    assert profile["changed_source_words"] > 0
    assert profile["changed_source_ratio"] > 0
    assert profile["shrink_ratio"] > 0


def test_v4_budget_lanes_are_ordered_from_conservative_to_aggressive():
    lanes = budget_lanes_for_unit(58, risk_score=0.8)

    assert [lane.lane_id for lane in lanes] == ["conservative", "route", "aggressive"]
    assert lanes[0].changed_source_ratio_max < lanes[1].changed_source_ratio_max < lanes[2].changed_source_ratio_max
    assert lanes[0].growth_ratio_max < lanes[1].growth_ratio_max


def test_v4_budget_profile_passes_lane_thresholds():
    lane = BudgetLane(
        lane_id="conservative",
        changed_source_ratio_max=0.25,
        growth_ratio_max=0.12,
        shrink_ratio_max=0.12,
        operation_families=("replace_only",),
        instruction="test",
    )

    assert budget_profile_passes(
        {"changed_source_ratio": 0.19, "growth_ratio": 0.0, "shrink_ratio": 0.034},
        lane,
    )
    assert not budget_profile_passes(
        {"changed_source_ratio": 0.31, "growth_ratio": 0.10, "shrink_ratio": 0.0},
        lane,
    )


def test_v4_layer3_safe_positive_is_not_always_material():
    row = _budget_lane_row({
        "ai_delta": 0.06,
        "topk_delta": 0.09,
        "external_delta": 0.285,
        "rank_delta": 0.035,
        "topk_calibrated_risk_delta": 0.245,
        "unsafe_word_ratio_delta": -0.184,
        "external_ai_flag_risk_delta": 0.059,
        "unsafe_cluster_delta": 0,
        "risky_window_delta": 0,
    })

    assert _is_safe_budget_lane_positive(row)
    assert not _is_material_budget_lane_positive(row)


def test_v4_layer3_material_positive_uses_blocker_movement():
    row = _budget_lane_row({
        "ai_delta": 0.12,
        "topk_delta": 0.2,
        "external_delta": 0.3,
        "rank_delta": 0.2,
        "topk_calibrated_risk_delta": 2.1,
        "unsafe_word_ratio_delta": 0.0,
        "external_ai_flag_risk_delta": 0.0,
        "unsafe_cluster_delta": 0,
        "risky_window_delta": 0,
    })

    assert _is_safe_budget_lane_positive(row)
    assert _is_material_budget_lane_positive(row)


def _budget_lane_row(scores):
    lane = BudgetLane(
        lane_id="conservative",
        changed_source_ratio_max=0.25,
        growth_ratio_max=0.12,
        shrink_ratio_max=0.12,
        operation_families=("replace_only",),
        instruction="test",
    )
    return {
        "apply_status": {"applied": True},
        "budget_lane": lane.to_dict(),
        "edit_budget_profile": {
            "changed_source_ratio": 0.1,
            "growth_ratio": 0.0,
            "shrink_ratio": 0.0,
        },
        "source_grounding": {"passed": True},
        "scores": scores,
        "candidate_goal": {
            "ai_footprint_gate": {
                "thresholds": {
                    "topk_calibrated_risk": 2.0,
                    "external_ai_flag_risk": 1.5,
                    "ai_likelihood": 1.0,
                }
            }
        },
    }
